import sys
import os
from loguru import logger

# 1. 确定日志保存的文件夹 (logs/)
current_dir = os.path.dirname(os.path.abspath(__file__)) # utils/
root_dir = os.path.dirname(current_dir) # baoshu_ai/
log_path = os.path.join(root_dir, "logs")

# 如果文件夹不存在，自动创建
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 2. 配置 Loguru
# 移除默认的控制台输出 (避免重复)
logger.remove()

# A. 输出到控制台 (屏幕上也能看到，带颜色)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# B. 输出到文件 (这是给以后查账用的)
# rotation="00:00": 每天零点自动创建一个新文件 (chat_2024-02-23.log)
# retention="10 days": 只保留最近 10 天的日志 (省硬盘空间)
# encoding="utf-8": 防止中文乱码
logger.add(
    os.path.join(log_path, "chat_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="10 days",
    encoding="utf-8",
    level="INFO",
    enqueue=True # 异步写入，防止阻塞主程序
)

# 导出这个配置好的 logger
__all__ = ["logger"]