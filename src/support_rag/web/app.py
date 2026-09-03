from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from support_rag.core.config import Settings
from support_rag.core.logging import configure_logging
from support_rag.web.config import WebSettings
from support_rag.web.models import Question
from support_rag.web.orchestrator import Orchestrator
from support_rag.web.store import ChatStore

STATIC = Path(__file__).with_name("static")
COOKIE = "rag_session"
DEMO_COOKIE = "rag_demo_session"
log = structlog.get_logger(__name__)


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class ChatQuestion(Question):
    use_history: bool = True


class RateLimiter:
    def __init__(self):
        self.events = OrderedDict()

    def allow(self, key, limit, window=60):
        now = time.monotonic()
        queue = self.events.setdefault(key, deque())
        self.events.move_to_end(key)
        while queue and queue[0] < now - window:
            queue.popleft()
        if len(self.events) > 4096:
            self.events.popitem(last=False)
        if len(queue) >= limit:
            return False
        queue.append(now)
        return True


def access_code(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "access-code"
    try:
        with path.open("x", encoding="utf-8") as stream:
            os.chmod(path, 0o600)
            stream.write(secrets.token_urlsafe(24) + "\n")
    except FileExistsError:
        pass
    code = path.read_text(encoding="utf-8").strip()
    if len(code) < 16:
        raise RuntimeError("Access code must contain at least 16 characters")
    return code


def create_app(*, web=None, orchestrator=None, password=None) -> FastAPI:
    web = web or WebSettings()
    allowed_origins = {web.public_url.rstrip("/"), "http://localhost:8080", "http://127.0.0.1:8080"}
    limiter = RateLimiter()
    gate = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        os.umask(0o077)
        settings = Settings()
        configure_logging(settings.log_level)
        app.state.store = ChatStore(web.state_dir / "chat.sqlite3")
        code = password or access_code(web.state_dir)
        app.state.password_hash = hashlib.sha256(code.encode()).digest()
        app.state.orchestrator = orchestrator or Orchestrator(settings, web)
        yield
        await app.state.orchestrator.close()

    app = FastAPI(
        title="Customer RAG",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def protection(request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if origin and origin not in allowed_origins:
                return JSONResponse({"detail": "Недопустимый источник запроса."}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "Cross-site request denied."}, status_code=403)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 16384:
                    return JSONResponse({"detail": "Слишком большой запрос."}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

    def browser_session(request: Request):
        names = (COOKIE, DEMO_COOKIE) if web.demo_mode else (COOKIE,)
        for name in names:
            record = app.state.store.session(request.cookies.get(name))
            if record is not None:
                return record
        return None

    def session_payload(record):
        return {
            "authenticated": record is not None,
            "csrf": record["csrf"] if record else None,
            "demo_mode": web.demo_mode,
        }

    def new_browser_session(request: Request, response: Response, name: str):
        token, record = app.state.store.new_session(web.session_seconds)
        response.set_cookie(
            name,
            token,
            max_age=web.session_seconds,
            httponly=True,
            samesite="lax",
            secure=(
                request.url.scheme == "https"
                or request.headers.get("origin", "").startswith("https://")
            ),
            path="/",
        )
        return session_payload(record)

    def session(request: Request):
        record = browser_session(request)
        if record is None:
            raise HTTPException(401, "Войдите, чтобы открыть чат.")
        if request.method not in {"GET", "HEAD"}:
            supplied = request.headers.get("x-csrf-token", "")
            if not hmac.compare_digest(record["csrf"], supplied):
                raise HTTPException(403, "Сессия устарела. Обновите страницу.")
        return record

    require_session = Depends(session)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/api/session")
    async def current_session(request: Request):
        return session_payload(browser_session(request))

    @app.post("/api/session")
    async def demo_session(request: Request, response: Response):
        if not web.demo_mode:
            raise HTTPException(404, "Демо-вход отключён.")
        existing = browser_session(request)
        if existing:
            return session_payload(existing)
        peer = request.client.host if request.client else "unknown"
        if not limiter.allow("session:" + peer, 10):
            raise HTTPException(429, "Слишком много новых сессий. Подождите минуту.")
        return new_browser_session(request, response, DEMO_COOKIE)

    @app.post("/api/login")
    async def login(body: Login, request: Request, response: Response):
        peer = request.client.host if request.client else "unknown"
        if not limiter.allow("login:" + peer, 10):
            raise HTTPException(429, "Слишком много попыток входа. Подождите минуту.")
        if not hmac.compare_digest(
            app.state.password_hash,
            hashlib.sha256(body.password.encode()).digest(),
        ):
            raise HTTPException(401, "Неверный код доступа.")
        existing = browser_session(request)
        if existing:
            return session_payload(existing)
        return new_browser_session(request, response, COOKIE)

    @app.post("/api/logout")
    async def logout(request: Request, response: Response, owner=require_session):
        for name in (COOKIE, DEMO_COOKIE):
            app.state.store.logout(request.cookies.get(name, ""))
            response.delete_cookie(name, path="/")
        return {"ok": True}

    @app.get("/api/health")
    async def health(owner=require_session):
        return await app.state.orchestrator.health()

    @app.get("/api/chats")
    async def chats(owner=require_session):
        return app.state.store.list_chats(owner["owner"])

    @app.post("/api/chats", status_code=201)
    async def new_chat(owner=require_session):
        try:
            return app.state.store.create_chat(owner["owner"], web.max_chats)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/chats/{chat_id}")
    async def get_chat(chat_id: str, owner=require_session):
        try:
            return app.state.store.get_chat(owner["owner"], chat_id)
        except KeyError as exc:
            raise HTTPException(404, "Чат не найден.") from exc

    @app.delete("/api/chats/{chat_id}")
    async def delete_chat(chat_id: str, owner=require_session):
        try:
            app.state.store.delete_chat(owner["owner"], chat_id)
        except KeyError as exc:
            raise HTTPException(404, "Чат не найден.") from exc
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/messages")
    async def send(chat_id: str, body: ChatQuestion, owner=require_session):
        await get_chat(chat_id, owner)
        if not limiter.allow("ask:" + owner["owner"], 20):
            raise HTTPException(429, "Лимит запросов. Подождите минуту.")
        try:
            await asyncio.wait_for(gate.acquire(), timeout=web.queue_timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                429, "Модель занята. Повторите запрос через несколько секунд."
            ) from exc
        try:
            chat = await get_chat(chat_id, owner)
            if len(chat["messages"]) >= web.max_turns * 2:
                raise HTTPException(409, "Создайте новый чат: достигнут лимит сообщений.")
            previous = [m["content"] for m in chat["messages"] if m["role"] == "user"]
            draft = await asyncio.wait_for(
                app.state.orchestrator.ask(body.question, previous if body.use_history else []),
                timeout=web.request_timeout_seconds,
            )
            return app.state.store.add_turn(
                owner["owner"],
                chat_id,
                body.question,
                draft,
                web.max_turns,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 422:
                raise HTTPException(422, "Вопрос слишком длинный. Сократите формулировку.") from exc
            log.warning("upstream_unavailable", status=exc.response.status_code)
            raise HTTPException(503, "Сервис временно недоступен. Попробуйте ещё раз.") from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "Модель не успела ответить. Попробуйте ещё раз.") from exc
        except (httpx.HTTPError, RuntimeError) as exc:
            log.warning("upstream_unavailable", error_type=type(exc).__name__)
            raise HTTPException(503, "Сервис временно недоступен. Попробуйте ещё раз.") from exc
        except KeyError as exc:
            raise HTTPException(404, "Чат удалён.") from exc
        finally:
            gate.release()

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
