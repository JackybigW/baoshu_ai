import asyncio
import os
import uuid
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
from agent_graph import app


class ChatRequest(BaseModel):
    message: str = ""
    session_id: Optional[str] = None


# 允许的跨域来源
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

api = FastAPI()
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

wecom_redis = aioredis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
crypto = WeComCrypto(WECOM_TOKEN, WECOM_AES_KEY, WECOM_CORPID)
wecom_api = WeComAPI(WECOM_CORPID, WECOM_SECRET, WECOM_KF_ID, wecom_redis)


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
        return {"response": output["messages"][-1].content, "session_id": session_id}

    # 2. 直接入队，不再拦截插嘴
    await buffer.add_message(session_id, message_text)

    # 3. 轮询等待领取任务权 (或被合并)
    max_wait = 25  # 考虑到排队，等待时间稍微延长到 25 秒
    start_time = asyncio.get_event_loop().time()

    while (asyncio.get_event_loop().time() - start_time) < max_wait:
        combined_text = await buffer.get_merged_message(session_id)

        if combined_text:
            # 我是这一波消息的执行者（Winner）
            try:
                logger.info(f"🚀 AI Winner Processing ({session_id}): {combined_text}")
                config = {"configurable": {"thread_id": session_id}}
                inputs = {"messages": [HumanMessage(content=combined_text)]}

                output = await asyncio.to_thread(app.invoke, inputs, config=config)

                all_messages = output.get("messages", [])
                ai_contents = []
                last_human_index = -1
                for i in range(len(all_messages) - 1, -1, -1):
                    if isinstance(all_messages[i], HumanMessage):
                        last_human_index = i
                        break
                for i in range(last_human_index + 1, len(all_messages)):
                    msg = all_messages[i]
                    if isinstance(msg, AIMessage) and msg.content:
                        ai_contents.append(msg.content.strip())

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

# ==========================================
# 企业微信客服 - 回调接口
# ==========================================


@api.get("/api/wecom/callback")
async def wecom_verify_url(msg_signature: str, timestamp: str, nonce: str, echostr: str):
    """
    企微后台保存「接收消息服务器 URL」时发来的 GET 探活验证请求。
    签名参与元素: [Token, timestamp, nonce, echostr]
    通过后必须返回 AES 解密后的纯明文。
    """
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
    企微客服消息/事件的 POST 回调。
    核心铁律：50ms 内返回空字符串，否则触发 5 秒超时重试风暴。
    """
    body = await request.body()

    try:
        # 1. 先解析外层 XML，提取 <Encrypt> 密文
        xml_tree = etree.fromstring(body)
        encrypt_text = xml_tree.find("Encrypt").text

        # 2. 用 Encrypt 字段做签名校验（P0 修复：不是 raw body）
        if not crypto.verify_signature(msg_signature, timestamp, nonce, encrypt_text):
            logger.warning("❌ WeCom POST signature verification FAILED")
            return PlainTextResponse(content="")

        # 3. AES 解密 → 内层 XML
        xml_content = crypto.decrypt(encrypt_text)
        decrypted_tree = etree.fromstring(xml_content.encode("utf-8"))

        event_type = decrypted_tree.findtext("Event", default="")

        if event_type == "kf_msg_or_event":
            sync_token = decrypted_tree.findtext("Token", default="")

            # 4. Redis SETNX 防抖：拦截企微 5 秒超时重试
            lock_key = f"wecom:lock:{sync_token}"
            is_new = await wecom_redis.setnx(lock_key, "1")
            if is_new:
                await wecom_redis.expire(lock_key, 300)
                # 5. 派发后台异步任务，立即释放 HTTP 连接
                background_tasks.add_task(_wecom_background_worker, sync_token)
                logger.info(f"📩 WeCom event dispatched (token={sync_token[:8]}...)")
            else:
                logger.info(f"🔁 WeCom duplicate event skipped (token={sync_token[:8]}...)")

    except Exception as e:
        logger.error(f"❌ WeCom callback parsing error: {e}")

    # 铁律：不论发生什么，都必须返回空字符串
    return PlainTextResponse(content="")


async def _wecom_background_worker(sync_token: str):
    """
    后台异步 Worker：
    1. 用 sync_token 拉取真实的 JSON 消息 (sync_msg)
    2. 送入 LangGraph 推理
    3. 用 send_msg 推送 AI 回复
    """
    logger.info(f"🔧 [Worker START] sync_token={sync_token[:12]}...")
    try:
        msg_data = await wecom_api.sync_wecom_messages(sync_token)
        logger.info(f"🔧 [Worker] sync_msg returned keys: {list(msg_data.keys())}")

        msg_list = msg_data.get("msg_list", [])
        is_bootstrap = msg_data.get("_is_bootstrap", False)

        if not msg_list:
            logger.info("🔧 [Worker] msg_list is empty, nothing to process")
            return

        # On bootstrap: sync_msg returns ALL history. Only process the LAST
        # user message (the one that just triggered this webhook).
        if is_bootstrap and len(msg_list) > 1:
            last_user_msg = None
            for m in reversed(msg_list):
                if m.get("origin") == 3 and m.get("msgtype") == "text":
                    last_user_msg = m
                    break
            if last_user_msg:
                logger.info(f"🔧 [Worker] Bootstrap: skipping {len(msg_list) - 1} "
                            f"historical msgs, processing only the latest")
                msg_list = [last_user_msg]
            else:
                logger.info("🔧 [Worker] Bootstrap: no user text msg found, skipping all")
                return

        for idx, msg in enumerate(msg_list):
            origin = msg.get("origin")
            msgtype = msg.get("msgtype")
            logger.info(f"🔧 [Worker] msg[{idx}]: origin={origin}, msgtype={msgtype}")

            # origin=3 表示消息来自外部微信客户
            if origin == 3 and msgtype == "text":
                ext_userid = msg.get("external_userid", "")
                user_text = msg.get("text", {}).get("content", "").strip()
                if not ext_userid or not user_text:
                    logger.info("🔧 [Worker] Skipping: empty userid or text")
                    continue

                # 检查是否已转人工
                if await wecom_redis.get(f"state:{ext_userid}:human_transferred"):
                    logger.info(f"🚫 Skipping AI for {ext_userid}: already transferred to human")
                    continue

                logger.info(f"🚀 WeCom Processing ({ext_userid}): {user_text}")

                # 调用 LangGraph（与 /chat 复用同一个 graph）
                config = {"configurable": {"thread_id": ext_userid}}
                inputs = {"messages": [HumanMessage(content=user_text)]}
                output = await asyncio.to_thread(app.invoke, inputs, config=config)

                # 如果 LangGraph 判定转人工，设置 Redis 隔离标记
                if output.get("dialog_status") == "FINISHED":
                    await wecom_redis.set(
                        f"state:{ext_userid}:human_transferred", "1", ex=86400  # 24h
                    )
                    logger.info(f"🔒 Human transfer flag set for {ext_userid}")

                # 提取 AI 回复
                all_messages = output.get("messages", [])
                ai_contents = []
                last_human_index = -1
                for i in range(len(all_messages) - 1, -1, -1):
                    if isinstance(all_messages[i], HumanMessage):
                        last_human_index = i
                        break
                for i in range(last_human_index + 1, len(all_messages)):
                    m = all_messages[i]
                    if isinstance(m, AIMessage) and m.content:
                        ai_contents.append(m.content.strip())

                logger.info(f"🔧 [Worker] AI reply segments: {len(ai_contents)}")

                if ai_contents:
                    for segment in ai_contents:
                        if segment:
                            result = await wecom_api.send_wecom_message(ext_userid, segment)
                            logger.info(f"📤 send_msg result: {result}")
                            await asyncio.sleep(0.5)
                else:
                    result = await wecom_api.send_wecom_message(ext_userid, "您好，请问有什么可以帮您？")
                    logger.info(f"📤 send_msg fallback result: {result}")
            else:
                logger.info(f"🔧 [Worker] Skipping msg: origin={origin}, msgtype={msgtype}")

    except Exception as e:
        logger.error(f"❌ WeCom background worker error: {e}", exc_info=True)

    logger.info(f"🔧 [Worker END] sync_token={sync_token[:12]}...")


api.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(api, host="0.0.0.0", port=8000)
