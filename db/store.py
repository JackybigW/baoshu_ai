import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

import asyncpg
from pydantic import BaseModel

from utils.logger import logger


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _text_ready(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    return str(value)


class DatabaseStore:
    def __init__(self) -> None:
        self.database_url = ""
        self.enabled = False
        self.pool: Optional[asyncpg.Pool] = None
        self.schema_path = Path(__file__).with_name("schema.sql")

    async def connect(self) -> None:
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        self.enabled = bool(self.database_url)
        if not self.enabled:
            logger.info("🗄️ DATABASE_URL 未配置，跳过 Postgres 初始化")
            return

        if self.pool is not None:
            return

        max_size = int(os.getenv("POSTGRES_POOL_MAX", "5"))
        min_size = int(os.getenv("POSTGRES_POOL_MIN", "1"))
        self.pool = await asyncpg.create_pool(
            dsn=self.database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
            server_settings={"application_name": "baoshu_ai"},
        )
        await self.init_schema()
        logger.info("🗄️ Postgres 连接成功，Schema 已就绪")

    async def close(self) -> None:
        if self.pool is None:
            return
        await self.pool.close()
        self.pool = None
        logger.info("🗄️ Postgres 连接已关闭")

    async def init_schema(self) -> None:
        if self.pool is None:
            return

        schema_sql = self.schema_path.read_text(encoding="utf-8")
        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

    async def persist_turn(
        self,
        *,
        channel: str,
        session_key: str,
        user_messages: Iterable[str],
        ai_messages: Iterable[str],
        output_state: Optional[dict[str, Any]] = None,
        external_userid: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return

        if self.pool is None:
            await self.connect()
        if self.pool is None:
            return

        state = output_state or {}
        profile_payload = _json_ready(state.get("profile")) or {}
        profile_json = json.dumps(profile_payload, ensure_ascii=False)
        dialog_status = _text_ready(state.get("dialog_status"))
        last_intent = _text_ready(state.get("last_intent"))

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                conversation_id = await conn.fetchval(
                    """
                    INSERT INTO conversations (
                        session_key,
                        channel,
                        external_userid,
                        dialog_status,
                        last_intent,
                        profile_json,
                        updated_at,
                        last_active_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW(), NOW())
                    ON CONFLICT (session_key) DO UPDATE
                    SET
                        channel = EXCLUDED.channel,
                        external_userid = COALESCE(EXCLUDED.external_userid, conversations.external_userid),
                        dialog_status = COALESCE(EXCLUDED.dialog_status, conversations.dialog_status),
                        last_intent = COALESCE(EXCLUDED.last_intent, conversations.last_intent),
                        profile_json = CASE
                            WHEN EXCLUDED.profile_json = '{}'::jsonb THEN conversations.profile_json
                            ELSE EXCLUDED.profile_json
                        END,
                        updated_at = NOW(),
                        last_active_at = NOW()
                    RETURNING id
                    """,
                    session_key,
                    channel,
                    external_userid,
                    dialog_status,
                    last_intent,
                    profile_json,
                )

                rows: list[tuple[int, str, str, str]] = []
                for content in user_messages:
                    text = content.strip()
                    if text:
                        rows.append((conversation_id, "human", text, "{}"))
                for content in ai_messages:
                    text = content.strip()
                    if text:
                        rows.append((conversation_id, "ai", text, "{}"))

                if rows:
                    await conn.executemany(
                        """
                        INSERT INTO messages (conversation_id, role, content, metadata_json)
                        VALUES ($1, $2, $3, $4::jsonb)
                        """,
                        rows,
                    )

    async def fetch_transcript(self, session_key: str) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        if self.pool is None:
            await self.connect()
        if self.pool is None:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT m.id, m.role, m.content, m.created_at
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE c.session_key = $1
                ORDER BY m.id ASC
                """,
                session_key,
            )
        return [dict(row) for row in rows]

    async def has_transcript(self, session_key: str) -> bool:
        if not self.enabled:
            return False

        if self.pool is None:
            await self.connect()
        if self.pool is None:
            return False

        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(1)
                FROM messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE c.session_key = $1
                """,
                session_key,
            )
        return bool(count)
