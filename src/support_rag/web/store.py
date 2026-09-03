from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class ChatStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    csrf TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    created REAL NOT NULL, updated REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chats_owner ON chats(owner, updated DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, payload TEXT,
                    created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def token_hash(token):
        return hashlib.sha256(token.encode()).hexdigest()

    def new_session(self, lifetime):
        token = secrets.token_urlsafe(32)
        owner, csrf = uuid4().hex, secrets.token_urlsafe(32)
        with self.connection() as db:
            db.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?)",
                (self.token_hash(token), owner, csrf, time.time() + lifetime),
            )
        return token, {"owner": owner, "csrf": csrf}

    def session(self, token):
        if not token or len(token) > 200:
            return None
        with self.connection() as db:
            row = db.execute(
                "SELECT owner, csrf FROM sessions WHERE token_hash=? AND expires>?",
                (self.token_hash(token), time.time()),
            ).fetchone()
        return dict(row) if row else None

    def logout(self, token):
        with self.connection() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (self.token_hash(token),))

    def create_chat(self, owner, limit):
        with self.connection() as db:
            count = db.execute("SELECT count(*) FROM chats WHERE owner=?", (owner,)).fetchone()[0]
            if count >= limit:
                raise ValueError("Достигнут лимит чатов. Удалите ненужный чат.")
            now = time.time()
            chat_id = uuid4().hex
            db.execute(
                "INSERT INTO chats VALUES (?, ?, ?, ?, ?)",
                (chat_id, owner, "Новый диалог", now, now),
            )
        return {"id": chat_id, "title": "Новый диалог", "created": now, "updated": now}

    def list_chats(self, owner):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id, title, created, updated FROM chats "
                    "WHERE owner=? ORDER BY updated DESC",
                    (owner,),
                )
            ]

    def get_chat(self, owner, chat_id):
        with self.connection() as db:
            row = db.execute(
                "SELECT id, title, created, updated FROM chats WHERE owner=? AND id=?",
                (owner, chat_id),
            ).fetchone()
            if not row:
                raise KeyError(chat_id)
            messages = []
            for message in db.execute(
                "SELECT id, role, content, payload, created FROM messages "
                "WHERE chat_id=? ORDER BY id",
                (chat_id,),
            ):
                item = dict(message)
                item["payload"] = json.loads(item["payload"]) if item["payload"] else None
                messages.append(item)
            return {**dict(row), "messages": messages}

    def delete_chat(self, owner, chat_id):
        with self.connection() as db:
            if not db.execute(
                "DELETE FROM chats WHERE owner=? AND id=?", (owner, chat_id)
            ).rowcount:
                raise KeyError(chat_id)

    def add_turn(self, owner, chat_id, question, draft, max_turns):
        with self.connection() as db:
            chat = db.execute(
                "SELECT id FROM chats WHERE owner=? AND id=?", (owner, chat_id)
            ).fetchone()
            if not chat:
                raise KeyError(chat_id)
            count = db.execute(
                "SELECT count(*) FROM messages WHERE chat_id=?", (chat_id,)
            ).fetchone()[0]
            if count >= max_turns * 2:
                raise ValueError("Диалог достиг лимита сообщений. Создайте новый чат.")
            now = time.time()
            db.execute(
                "INSERT INTO messages(chat_id,role,content,created) VALUES(?,?,?,?)",
                (chat_id, "user", question, now),
            )
            db.execute(
                "INSERT INTO messages(chat_id,role,content,payload,created) VALUES(?,?,?,?,?)",
                (chat_id, "assistant", draft.response.answer, draft.model_dump_json(), now),
            )
            if count == 0:
                db.execute(
                    "UPDATE chats SET title=?, updated=? WHERE id=?", (question[:70], now, chat_id)
                )
            else:
                db.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))
        return self.get_chat(owner, chat_id)
