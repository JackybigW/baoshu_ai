import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as aioredis
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from lxml import etree
from pydantic import BaseModel

from utils.buffer import MessageBuffer
from utils.logger import logger
from utils.wecom_api import WeComAPI
from utils.wecom_crypto import WeComCrypto

_ = load_dotenv(find_dotenv(), override=True)
from db import db_store
from agent_graph import app


class ChatRequest(BaseModel):
    message: str = ""
    session_id: Optional[str] = None


# 允许的跨域来源
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

@asynccontextmanager
async def lifespan(_: FastAPI):
    await db_store.connect()
    try:
        yield
    finally:
        await db_store.close()


api = FastAPI(lifespan=lifespan)
api.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2.5s 仍然是黄金等待期
buffer = MessageBuffer(wait_time=2.5)

# ==========================================
# 企业微信客服 - 初始化
# ==========================================
WECOM_CORPID = os.getenv("WECOM_CORPID", "")
WECOM_SECRET = os.getenv("WECOM_SECRET", "")
WECOM_TOKEN = os.getenv("WECOM_TOKEN", "")
WECOM_AES_KEY = os.getenv("WECOM_AES_KEY", "")
WECOM_KF_ID = os.getenv("WECOM_KF_ID", "")

# 真人感参数（可通过 .env 调整）
WECOM_BUFFER_WAIT = float(os.getenv("WECOM_BUFFER_WAIT", "3.5"))
WECOM_TYPING_SPEED = float(os.getenv("WECOM_TYPING_SPEED", "0.15"))
WECOM_TYPING_MAX = float(os.getenv("WECOM_TYPING_MAX", "8.0"))

wecom_redis = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
crypto = WeComCrypto(WECOM_TOKEN, WECOM_AES_KEY, WECOM_CORPID)
wecom_api = WeComAPI(WECOM_CORPID, WECOM_SECRET, WECOM_KF_ID, wecom_redis)

# 企微消息缓冲 debounce 任务表 {ext_userid: asyncio.Task}
_wecom_debounce_tasks: dict = {}


def _extract_ai_contents(all_messages) -> list[str]:
    ai_contents = []
    last_human_index = -1
    for i in range(len(all_messages) - 1, -1, -1):
        if isinstance(all_messages[i], HumanMessage):
            last_human_index = i
            break

    for i in range(last_human_index + 1, len(all_messages)):
        message = all_messages[i]
        if isinstance(message, AIMessage) and message.content:
            ai_contents.append(message.content.strip())
    return ai_contents


async def _persist_turn(
    *,
    channel: str,
    session_key: str,
    user_messages: list[str],
    ai_messages: list[str],
    output_state: dict,
    external_userid: Optional[str] = None,
):
    try:
        await db_store.persist_turn(
            channel=channel,
            session_key=session_key,
            user_messages=user_messages,
            ai_messages=ai_messages,
            output_state=output_state,
            external_userid=external_userid,
        )
    except Exception as exc:
        logger.error(f"❌ Postgres persist error: {exc}", exc_info=True)


@api.get("/api/wecom/callback")
async def wecom_verify_url(
    msg_signature: str, timestamp: str, nonce: str, echostr: str
):
    """企微 GET 探活验证。"""
    if crypto.verify_signature(msg_signature, timestamp, nonce, echostr):
        decrypted_echo = crypto.decrypt(echostr)
        logger.info("✅ WeCom URL verification passed")
        return PlainTextResponse(content=decrypted_echo)
    logger.warning("❌ WeCom URL verification FAILED")
    return PlainTextResponse(content="signature error", status_code=403)


@api.post("/api/wecom/callback")
async def wecom_receive_event(
    request: Request,
    msg_signature: str,
    timestamp: str,
    nonce: str,
    background_tasks: BackgroundTasks
):
    """
    企微客服 POST 回调。50ms 内必须返回空字符串。
    新逻辑：sync_msg → 提取用户消息 → 存入 Redis buffer → debounce 合并
    """
    body = await request.body()

    try:
        xml_tree = etree.fromstring(body)
        encrypt_text = xml_tree.find("Encrypt").text

        if not crypto.verify_signature(msg_signature, timestamp, nonce, encrypt_text):
            logger.warning("❌ WeCom POST signature verification FAILED")
            return PlainTextResponse(content="")

        xml_content = crypto.decrypt(encrypt_text)
        decrypted_tree = etree.fromstring(xml_content.encode("utf-8"))
        event_type = decrypted_tree.findtext("Event", default="")

        if event_type == "kf_msg_or_event":
            sync_token = decrypted_tree.findtext("Token", default="")

            lock_key = f"wecom:lock:{sync_token}"
            is_new = await wecom_redis.setnx(lock_key, "1")
            if is_new:
                await wecom_redis.expire(lock_key, 300)
                background_tasks.add_task(_wecom_fetch_and_buffer, sync_token)
                logger.info(f"📩 WeCom event dispatched (token={sync_token[:8]}...)")
            else:
                logger.info(f"🔁 WeCom duplicate event skipped (token={sync_token[:8]}...)")

    except Exception as e:
        logger.error(f"❌ WeCom callback parsing error: {e}")

    return PlainTextResponse(content="")


async def _wecom_fetch_and_buffer(sync_token: str):
    """
    Step 1: sync_msg 拉取用户消息
    Step 2: 存入 Redis buffer + 启动/重置 debounce 定时器
    不做任何 LangGraph 推理，只负责"收"。
    """
    try:
        msg_data = await wecom_api.sync_wecom_messages(sync_token)
        msg_list = msg_data.get("msg_list", [])
        is_bootstrap = msg_data.get("_is_bootstrap", False)

        if not msg_list:
            return

        # Bootstrap: 只取最后一条用户消息
        if is_bootstrap and len(msg_list) > 1:
            last_user_msg = None
            for m in reversed(msg_list):
                if m.get("origin") == 3 and m.get("msgtype") == "text":
                    last_user_msg = m
                    break
            if last_user_msg:
                logger.info(
                    f"🔧 Bootstrap: skipping {len(msg_list) - 1} historical msgs"
                )
                msg_list = [last_user_msg]
            else:
                return

        for msg in msg_list:
            origin = msg.get("origin")
            msgtype = msg.get("msgtype")
            ext_userid = msg.get("external_userid", "")
            event = msg.get("event", {})
            
            # 兼容：有些 event 类型的 external_userid 在 event 字典里
            if not ext_userid and event:
                ext_userid = event.get("external_userid", "")
            
            user_text = None
            
            if origin == 3 and msgtype == "text":
                user_text = msg.get("text", {}).get("content", "").strip()
            # 处理用户进入客服会话事件，触发打招呼逻辑
            elif msgtype == "event" and event.get("event_type") == "enter_session":
                if not ext_userid:
                    continue
                if await wecom_redis.get(f"state:{ext_userid}:human_transferred"):
                    logger.info(f"🚫 Skipping welcome AI for {ext_userid}: already transferred")
                    continue

                welcome_code = event.get("welcome_code", "")
                await _wecom_handle_enter_session(ext_userid, welcome_code)
                continue
                
            if not ext_userid or user_text is None:
                continue

            if await wecom_redis.get(f"state:{ext_userid}:human_transferred"):
                logger.info(f"🚫 Skipping AI for {ext_userid}: already transferred")
                continue

            # 存入 Redis buffer
            buf_key = f"wecom:buffer:{ext_userid}"
            await wecom_redis.rpush(buf_key, user_text)
            await wecom_redis.expire(buf_key, 600)
            logger.info(f"📥 Buffered ({ext_userid}): {user_text if user_text else '[ENTER_SESSION]'} ")

            # 启动/重置 debounce 定时器
            _reset_debounce_timer(ext_userid)

    except Exception as e:
        logger.error(f"❌ WeCom fetch_and_buffer error: {e}", exc_info=True)


async def _wecom_handle_enter_session(ext_userid: str, welcome_code: str):
    """
    enter_session 不能走普通 send_msg。
    必须使用事件欢迎语接口，否则会返回 95018。
    """
    if not welcome_code:
        logger.warning(f"⚠️ Missing welcome_code for enter_session: {ext_userid}")
        return

    process_lock = f"wecom:process_lock:{ext_userid}"
    acquired = await wecom_redis.set(process_lock, "1", nx=True, ex=60)
    if not acquired:
        logger.info(f"🔁 Welcome message skipped due to active worker: {ext_userid}")
        return

    try:
        logger.info(f"👋 Handling enter_session welcome for {ext_userid}")
        config = {"configurable": {"thread_id": ext_userid}}
        output = await asyncio.to_thread(app.invoke, {"messages": []}, config=config)

        all_messages = output.get("messages", [])
        ai_contents = _extract_ai_contents(all_messages)
        welcome_text = "\n".join(text for text in ai_contents if text).strip()
        if not welcome_text:
            welcome_text = "您好，欢迎咨询！ 跟上暴叔的节奏～"

        result = await wecom_api.send_wecom_welcome_message(welcome_code, welcome_text)
        logger.info(f"📤 send_msg_on_event result: {result}")

        if result.get("errcode") == 0:
            await _persist_turn(
                channel="wecom",
                session_key=ext_userid,
                user_messages=[],
                ai_messages=[welcome_text],
                output_state=output,
                external_userid=ext_userid,
            )
    except Exception as e:
        logger.error(f"❌ WeCom enter_session welcome error: {e}", exc_info=True)
    finally:
        await wecom_redis.delete(process_lock)


def _reset_debounce_timer(ext_userid: str):
    """取消旧定时器，启动新的 N 秒倒计时。每条新消息重置。"""
    global _wecom_debounce_tasks
    if ext_userid in _wecom_debounce_tasks:
        _wecom_debounce_tasks[ext_userid].cancel()

    _wecom_debounce_tasks[ext_userid] = asyncio.create_task(
        _wecom_debounce_fire(ext_userid)
    )


async def _wecom_debounce_fire(ext_userid: str):
    """
    等用户停顿 WECOM_BUFFER_WAIT 秒后触发：
    合并 buffered 消息 → LangGraph → 发送 AI 回复（动态打字延迟）
    """
    try:
        await asyncio.sleep(WECOM_BUFFER_WAIT)
    except asyncio.CancelledError:
        return  # 被新消息重置了

    # Redis 分布式锁防并发处理
    process_lock = f"wecom:process_lock:{ext_userid}"
    acquired = await wecom_redis.set(process_lock, "1", nx=True, ex=180)
    if not acquired:
        logger.info(f"🔁 Another worker already processing {ext_userid}")
        return

    try:
        # 1. 取走所有 buffered 消息
        buf_key = f"wecom:buffer:{ext_userid}"
        messages = await wecom_redis.lrange(buf_key, 0, -1)
        await wecom_redis.delete(buf_key)

        if not messages:
            return

        combined_text = "\n".join(m for m in messages if m).strip()
        logger.info(
            f"🚀 WeCom Processing ({ext_userid}): "
            f"{len(messages)} msgs merged → {combined_text[:60] if combined_text else '[ENTER_SESSION]'}..."
        )

        # 2. 调用 LangGraph (复用 /chat 开场白逻辑)
        config = {"configurable": {"thread_id": ext_userid}}
        if not combined_text:
            inputs = {"messages": []}
        else:
            inputs = {"messages": [HumanMessage(content=combined_text)]}
            
        output = await asyncio.to_thread(app.invoke, inputs, config=config)

        # 3. 如果 LangGraph 判定转人工
        if output.get("dialog_status") == "FINISHED":
            await wecom_redis.set(
                f"state:{ext_userid}:human_transferred", "1", ex=86400
            )
            logger.info(f"🔒 Human transfer flag set for {ext_userid}")

        # 4. 提取 AI 回复
        all_messages = output.get("messages", [])
        ai_contents = _extract_ai_contents(all_messages)
        persisted_ai_messages = ai_contents or ["您好，请问有什么可以帮您？"]
        persisted_user_messages = [msg.strip() for msg in messages if msg.strip()]
        await _persist_turn(
            channel="wecom",
            session_key=ext_userid,
            user_messages=persisted_user_messages,
            ai_messages=persisted_ai_messages,
            output_state=output,
            external_userid=ext_userid,
        )

        # 5. 发送 AI 回复（动态打字延迟）
        if ai_contents:
            for idx, segment in enumerate(ai_contents):
                if segment:
                    result = await wecom_api.send_wecom_message(ext_userid, segment)
                    logger.info(f"📤 send_msg result: {result}")
                    # 非最后一段：动态间隔模拟真人打字
                    if idx < len(ai_contents) - 1:
                        next_segment = ai_contents[idx + 1]
                        delay = min(
                            max(len(next_segment) * WECOM_TYPING_SPEED, 2.0), WECOM_TYPING_MAX
                        )
                        logger.info(
                            f"⏱️ Typing delay: {delay:.1f}s "
                            f"for next {len(next_segment)} chars"
                        )
                        await asyncio.sleep(delay)
        else:
            await wecom_api.send_wecom_message(ext_userid, "您好，请问有什么可以帮您？")

    except Exception as e:
        logger.error(f"❌ WeCom debounce worker error: {e}", exc_info=True)
    finally:
        await wecom_redis.delete(process_lock)
        _wecom_debounce_tasks.pop(ext_userid, None)

        # 检查 AI 打字期间，用户是否又发了新消息 (插嘴)
        # 如果有，立刻重新触发 debounce，保证新消息不被卡死
        leftover = await wecom_redis.llen(f"wecom:buffer:{ext_userid}")
        if leftover > 0:
            logger.info(f"🔄 Interleaved messages detected for {ext_userid} ({leftover} queued). Retriggering...")
            _reset_debounce_timer(ext_userid)

    logger.info(f"🔧 [Worker END] {ext_userid}")





@api.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.session_id:
        req.session_id = str(uuid.uuid4())

    session_id = req.session_id
    message_text = req.message.strip()

    # 1. 开场白逻辑
    if not message_text:
        config = {"configurable": {"thread_id": session_id}}
        output = await asyncio.to_thread(app.invoke, {"messages": []}, config=config)
        response_text = output["messages"][-1].content
        await _persist_turn(
            channel="web",
            session_key=session_id,
            user_messages=[],
            ai_messages=[response_text],
            output_state=output,
        )
        return {"response": response_text, "session_id": session_id}

    # 2. 直接入队，不再拦截插嘴
    await buffer.add_message(session_id, message_text)

    # 3. 轮询等待领取任务权 (或被合并)
    max_wait = 25  # 考虑到排队，等待时间稍微延长到 25 秒
    start_time = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - start_time) < max_wait:
        batch = await buffer.get_message_batch(session_id)

        if batch:
            # 我是这一波消息的执行者（Winner）
            try:
                raw_user_messages = [msg.strip() for msg in batch["messages"] if msg.strip()]
                combined_text = batch["combined_text"].strip()
                logger.info(f"🚀 AI Winner Processing ({session_id}): {combined_text}")
                config = {"configurable": {"thread_id": session_id}}
                inputs = {"messages": [HumanMessage(content=combined_text)]}

                output = await asyncio.to_thread(app.invoke, inputs, config=config)

                all_messages = output.get("messages", [])
                ai_contents = _extract_ai_contents(all_messages)
                persisted_ai_messages = ai_contents or ["好的。"]
                await _persist_turn(
                    channel="web",
                    session_key=session_id,
                    user_messages=raw_user_messages,
                    ai_messages=persisted_ai_messages,
                    output_state=output,
                )

                response_text = "|||".join(ai_contents) if ai_contents else "好的。"
                await buffer.release_lock(session_id)  # AI 跑完，释放锁
                return {"response": response_text, "session_id": session_id}

            except Exception as e:
                await buffer.release_lock(session_id)
                logger.error(f"❌ AI Error: {str(e)}")
                raise HTTPException(status_code=500, detail="系统故障。")

        # 这里的检查依然重要：如果我发现缓冲区空了且锁被释放了，
        # 说明我的消息已经被同一波的某个 Winner 领走并处理完了。
        buf_len = await buffer.redis.llen(f"buffer:{session_id}")
        lock_exists = await buffer.redis.get(f"lock:{session_id}")
        ready_exists = await buffer.redis.get(f"ready:{session_id}")
        if buf_len == 0 and not lock_exists and not ready_exists:
            # 安全退出：消息已被 Winner 领走
            return {"response": "__MERGED__", "session_id": session_id}

        await asyncio.sleep(0.5)

    return {"response": "__MERGED__", "session_id": session_id}


api.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
