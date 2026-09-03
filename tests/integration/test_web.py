from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from support_rag.core.schemas import RAGResponse, RAGSource
from support_rag.generation.prompt import generation_failure_message, insufficient_message
from support_rag.web.app import create_app
from support_rag.web.config import WebSettings
from support_rag.web.models import Draft
from support_rag.web.orchestrator import confidence_gate

CODE = "demo-access-code-for-tests-only"


def response():
    return RAGResponse(
        answer="Откройте настройки подписки. [SOURCE 1]",
        model="test",
        context_tokens=15,
        sources=[
            RAGSource(
                source_number=1,
                document_id="doc",
                chunk_id="doc:0",
                title="Статья",
                source_url="https://support.kaspersky.com/test",
                retrieval_source="dense",
                cited=True,
            )
        ],
        cited_source_numbers=[1],
        finish_reason="stop",
    )


class FakeOrchestrator:
    def __init__(self):
        self.calls = []
        self.failed = False

    async def close(self):
        pass

    async def health(self):
        return {"retrieval": True, "generation": True}

    async def ask(self, question, previous):
        self.calls.append((question, previous))
        if self.failed:
            raise httpx.ConnectError("secret-internal-host:8081")
        answer = response()
        return Draft(
            response=answer,
            confidence=confidence_gate(answer, question),
            retrieval_query=question,
            latency_seconds=0.1,
        )


@pytest.fixture
def configured(tmp_path):
    fake = FakeOrchestrator()
    web = WebSettings(state_dir=tmp_path, public_url="https://demo.example")
    return web, fake


def login(client):
    result = client.post("/api/login", json={"password": CODE})
    assert result.status_code == 200
    client.headers["X-CSRF-Token"] = result.json()["csrf"]
    return result


def test_auth_csrf_and_origin_are_enforced(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/chats").status_code == 401
        assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
        assert (
            client.post(
                "/api/login", json={"password": CODE}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        logged = login(client)
        assert "HttpOnly" in logged.headers["set-cookie"]
        assert client.post("/api/chats").status_code == 201
        del client.headers["X-CSRF-Token"]
        assert client.post("/api/chats").status_code == 403
        assert client.get("/api/chats").headers["cache-control"] == "no-store"


def test_history_and_sessions_survive_restart(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        result = client.post(f"/api/chats/{chat_id}/messages", json={"question": "Как отключить?"})
        assert result.status_code == 200
        assert len(result.json()["messages"]) == 2
        token = client.cookies.get("rag_session")
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        client.cookies.set("rag_session", token)
        assert client.get("/api/session").json()["authenticated"] is True
        assert len(client.get(f"/api/chats/{chat_id}").json()["messages"]) == 2


def test_sessions_cannot_read_delete_or_use_each_others_chats(configured):
    web, fake = configured
    app = create_app(web=web, orchestrator=fake, password=CODE)
    with TestClient(app) as first:
        login(first)
        chat_id = first.post("/api/chats").json()["id"]
        first.cookies.clear()
        login(first)
        assert first.get("/api/chats").json() == []
        assert first.get(f"/api/chats/{chat_id}").status_code == 404
        assert first.delete(f"/api/chats/{chat_id}").status_code == 404
        assert (
            first.post(f"/api/chats/{chat_id}/messages", json={"question": "Вопрос"}).status_code
            == 404
        )
        assert not fake.calls


def test_failed_request_does_not_create_half_a_turn_or_leak_backend(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        fake.failed = True
        result = client.post(f"/api/chats/{chat_id}/messages", json={"question": "Вопрос"})
        assert result.status_code == 503
        assert "secret-internal" not in result.text
        assert client.get(f"/api/chats/{chat_id}").json()["messages"] == []


def test_history_uses_only_user_questions_and_can_be_disabled(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        for question, use_history in [
            ("Первый вопрос", True),
            ("А потом?", True),
            ("Другой вопрос", False),
        ]:
            assert (
                client.post(
                    f"/api/chats/{chat_id}/messages",
                    json={
                        "question": question,
                        "use_history": use_history,
                    },
                ).status_code
                == 200
            )
        assert fake.calls[1][1] == ["Первый вопрос"]
        assert fake.calls[2][1] == []


@pytest.mark.parametrize("question", ["", " " * 5, "a" * 1001])
def test_invalid_question_is_rejected_before_generation(configured, question):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        assert (
            client.post(f"/api/chats/{chat_id}/messages", json={"question": question}).status_code
            == 422
        )
        assert not fake.calls


def test_large_body_and_delete(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        assert client.post("/api/login", content="a" * 17000).status_code == 413
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        assert client.delete(f"/api/chats/{chat_id}").status_code == 200
        assert client.get(f"/api/chats/{chat_id}").status_code == 404


@pytest.mark.parametrize(
    "change",
    [
        {"finish_reason": "length"},
        {"cited_source_numbers": []},
        {"invalid_citation_numbers": [99]},
        {"answer": "only english letters here"},
    ],
)
def test_confidence_gate_rejects_invalid_contract(change):
    answer = response().model_copy(update=change)
    confidence = confidence_gate(answer, "Как отключить подписку?")
    assert confidence.level == "insufficient"
    assert confidence.label == "Ответ не сформирован"
    assert answer.abstained is True
    assert answer.abstention_reason == "generation_contract"
    assert answer.answer == generation_failure_message("Russian")
    assert not answer.cited_source_numbers


@pytest.mark.parametrize(
    ("reason", "label", "message"),
    [
        ("insufficient_context", "Недостаточно данных", insufficient_message("Russian")),
        ("generation_contract", "Ответ не сформирован", generation_failure_message("Russian")),
    ],
)
def test_abstention_reason_is_distinguished_in_the_chat_payload(configured, reason, label, message):
    web, _ = configured

    class AbstainingOrchestrator(FakeOrchestrator):
        async def ask(self, question, previous):
            answer = RAGResponse(
                answer=message,
                model="test",
                context_tokens=15,
                abstained=True,
                abstention_reason=reason,
                finish_reason="stop",
            )
            return Draft(
                response=answer,
                confidence=confidence_gate(answer, question),
                retrieval_query=question,
                latency_seconds=0.1,
            )

    app = create_app(web=web, orchestrator=AbstainingOrchestrator(), password=CODE)
    with TestClient(app) as client:
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        result = client.post(f"/api/chats/{chat_id}/messages", json={"question": "Вопрос"})
        assert result.status_code == 200
        for chat in (result.json(), client.get(f"/api/chats/{chat_id}").json()):
            assistant = chat["messages"][-1]
            assert assistant["content"] == message
            assert assistant["payload"]["confidence"]["label"] == label
            assert assistant["payload"]["response"]["abstention_reason"] == reason


def test_confidence_does_not_claim_calibrated_probability():
    answer = response()
    assert confidence_gate(answer, "Вопрос").level == "supported"
    answer.generation_attempts = 2
    assert confidence_gate(answer, "Вопрос").level == "review"


def test_demo_entry_is_opt_in(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        assert client.get("/api/session").json()["demo_mode"] is False
        assert client.post("/api/session").status_code == 404
        assert client.get("/api/chats").status_code == 401


def test_demo_sessions_are_reused_private_and_csrf_protected(configured):
    web, fake = configured
    web = web.model_copy(update={"demo_mode": True})
    app = create_app(web=web, orchestrator=fake, password=CODE)
    with TestClient(app, base_url="https://demo.example") as client:
        assert client.get("/api/session").json()["authenticated"] is False
        assert not client.cookies
        assert (
            client.post("/api/session", headers={"Origin": "https://evil.example"}).status_code
            == 403
        )
        entry = client.post("/api/session")
        assert entry.status_code == 200
        assert "HttpOnly" in entry.headers["set-cookie"]
        assert "Secure" in entry.headers["set-cookie"]
        assert client.post("/api/chats").status_code == 403
        client.headers["X-CSRF-Token"] = entry.json()["csrf"]
        chat_id = client.post("/api/chats").json()["id"]
        repeated = client.post("/api/session")
        assert repeated.json()["csrf"] == entry.json()["csrf"]
        assert "set-cookie" not in repeated.headers
        client.cookies.clear()
        second = client.post("/api/session")
        client.headers["X-CSRF-Token"] = second.json()["csrf"]
        assert client.get("/api/chats").json() == []
        assert client.get(f"/api/chats/{chat_id}").status_code == 404
        assert client.delete(f"/api/chats/{chat_id}").status_code == 404
        assert (
            client.post(f"/api/chats/{chat_id}/messages", json={"question": "Вопрос"}).status_code
            == 404
        )
        assert not fake.calls


def test_guest_cannot_enter_when_demo_is_disabled_and_private_history_survives(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        login(client)
        chat_id = client.post("/api/chats").json()["id"]
        private_cookies = dict(client.cookies)
    demo = web.model_copy(update={"demo_mode": True})
    with TestClient(create_app(web=demo, orchestrator=fake, password=CODE)) as client:
        client.cookies.update(private_cookies)
        assert client.post("/api/session").status_code == 200
        assert client.get(f"/api/chats/{chat_id}").status_code == 200
        client.cookies.clear()
        assert client.post("/api/session").status_code == 200
        guest_cookies = dict(client.cookies)
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        client.cookies.update(guest_cookies)
        assert client.get("/api/session").json()["authenticated"] is False
        assert client.get("/api/chats").status_code == 401
        client.cookies.update(private_cookies)
        assert client.get(f"/api/chats/{chat_id}").status_code == 200


def test_new_demo_sessions_are_rate_limited(configured):
    web, fake = configured
    web = web.model_copy(update={"demo_mode": True})
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        for _ in range(10):
            client.cookies.clear()
            assert client.post("/api/session").status_code == 200
        assert client.post("/api/session").status_code == 200
        client.cookies.clear()
        assert client.post("/api/session").status_code == 429


def test_page_and_assets_revalidate_cached_responses(configured):
    web, fake = configured
    with TestClient(create_app(web=web, orchestrator=fake, password=CODE)) as client:
        for path in ("/", "/static/app.js", "/static/styles.css"):
            result = client.get(path)
            assert result.status_code == 200
            assert result.headers["cache-control"] == "no-cache"
        cached = client.get(
            "/static/app.js",
            headers={"If-None-Match": client.get("/static/app.js").headers["etag"]},
        )
        assert cached.status_code == 304
        assert cached.headers["cache-control"] == "no-cache"
