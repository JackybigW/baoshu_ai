# config/settings.py
from typing import List

# 触发低预算强制转换的负债关键词
DEBT_KEYWORDS: List[str] = [
    "贷", "负债", "欠款", "欠债", "还债", 
    "借钱", "还钱", "周转", "抵押", "上岸"
]

# 触发转人工强制转换的信号关键词
HANDOFF_SIGNALS: List[str] = [
    "语音", "电话", "通话", "电联", 
    "面谈", "面聊", "加个微", "微信聊", "手机号"
]

# 具有“粘性”（一旦进入就不轻易退出的）意图列表
STICKY_INTENTS: List[str] = [
    "ART_CONSULTING", 
    "HIGH_VALUE", 
    "LOW_BUDGET", 
    "SALES_READY"
]
