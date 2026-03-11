import asyncio
import uuid

import httpx

from utils.logger import logger


class WeComAPI:
    def __init__(self, corpid: str, secret: str, kf_id: str, redis_client):
        self.corpid = corpid
        self.secret = secret
        self.kf_id = kf_id
        self.redis = redis_client

    async def get_access_token(self) -> str:
        """
        Fetches the WeCom Access Token with Redis cache + distributed mutex lock.
        Prevents cache stampede (惊群效应) when token expires under high concurrency.
        """
        token = await self.redis.get("wecom:access_token")
        if token:
            return token

        # Distributed mutex: only one coroutine refreshes the token at a time
        lock_key = "wecom:token_refresh_lock"
        acquired = await self.redis.set(lock_key, "1", nx=True, ex=10)

        if acquired:
            # Winner: fetch new token from Tencent API
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    url = (
                        f"https://qyapi.weixin.qq.com/cgi-bin/gettoken"
                        f"?corpid={self.corpid}&corpsecret={self.secret}"
                    )
                    resp = await client.get(url)
                    data = resp.json()
                    if data.get("errcode") == 0:
                        new_token = data["access_token"]
                        await self.redis.set(
                            "wecom:access_token", new_token,
                            ex=data["expires_in"] - 200
                        )
                        return new_token
                    raise Exception(f"WeCom gettoken failed: {data}")
            finally:
                await self.redis.delete(lock_key)
        else:
            # Loser: spin-wait for winner to finish refreshing
            for _ in range(20):
                await asyncio.sleep(0.5)
                token = await self.redis.get("wecom:access_token")
                if token:
                    return token
            raise Exception("Timed out waiting for access token refresh")

    async def sync_wecom_messages(self, sync_token: str) -> dict:
        """
        Fetches NEW messages incrementally using Redis-stored cursor.
        - First call (no cursor): passes the webhook `token` to bootstrap
        - Subsequent calls: passes stored `cursor` to only get new messages
        After each call, stores `next_cursor` in Redis for the next round.
        """
        access_token = await self.get_access_token()

        # Build payload with cursor for incremental fetching
        cursor_key = f"wecom:sync_cursor:{self.kf_id}"
        stored_cursor = await self.redis.get(cursor_key)

        payload = {"open_kfid": self.kf_id, "limit": 1000}
        if stored_cursor:
            # Subsequent call: use stored cursor, skip the event token
            payload["cursor"] = stored_cursor
            logger.info(f"📡 sync_msg: using stored cursor={stored_cursor[:12]}...")
        else:
            # First call: use the webhook token to bootstrap
            payload["token"] = sync_token
            logger.info(f"📡 sync_msg: bootstrapping with token={sync_token[:12]}...")

        url = f"https://qyapi.weixin.qq.com/cgi-bin/kf/sync_msg?access_token={access_token}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()

            # Store next_cursor for future incremental pulls
            next_cursor = data.get("next_cursor")
            if next_cursor:
                await self.redis.set(cursor_key, next_cursor)

            msg_count = len(data.get("msg_list", []))
            logger.info(f"📡 sync_msg response: errcode={data.get('errcode')}, "
                        f"msg_count={msg_count}, has_next_cursor={bool(next_cursor)}")
            if data.get("errcode") != 0:
                logger.error(f"❌ sync_msg API error: {data}")

            # Flag for caller: is this a bootstrap (first) call?
            data["_is_bootstrap"] = not bool(stored_cursor)
            return data

    async def send_wecom_message(self, external_userid: str, text: str) -> dict:
        """
        Sends an asynchronous text reply to the user via WeCom send_msg API.
        """
        access_token = await self.get_access_token()
        url = f"https://qyapi.weixin.qq.com/cgi-bin/kf/send_msg?access_token={access_token}"
        payload = {
            "touser": external_userid,
            "open_kfid": self.kf_id,
            "msgtype": "text",
            "text": {"content": text}
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            result = resp.json()
            if result.get("errcode") != 0:
                logger.warning(f"send_msg error: {result}")
            return result

    async def send_wecom_welcome_message(self, welcome_code: str, text: str) -> dict:
        """
        Sends a text welcome message in response to enter_session.
        This uses the event-specific API instead of the normal send_msg API.
        """
        access_token = await self.get_access_token()
        url = f"https://qyapi.weixin.qq.com/cgi-bin/kf/send_msg_on_event?access_token={access_token}"
        payload = {
            "code": welcome_code,
            "msgid": uuid.uuid4().hex,
            "msgtype": "text",
            "text": {"content": text},
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            result = resp.json()
            if result.get("errcode") != 0:
                logger.warning(f"send_msg_on_event error: {result}")
            return result
