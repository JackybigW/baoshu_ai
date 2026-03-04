# -*- coding: utf-8 -*-
import os
import sys
from typing import List

# --- 1. 确保能导入项目中的模块 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 2. 导入组件 ---
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from state import AgentState, CustomerProfile, IntentType
from nodes.consultants import chit_chat_node, first_greeting_node
from dotenv import load_dotenv

# 加载环境变量 (需要 API Key)
load_dotenv()

def print_header(title: str):
    print(f"\n{'='*20} {title} {'='*20}")

def run_interactive_test():
    """
    交互式测试脚本：
    用于评估 chit_chat_node 或 first_greeting_node 的 Prompt 效果。
    """
    print_header("暴叔 AI - 闲聊/破冰 效果评估")
    print("提示: 输入 'exit' 退出测试，输入 'reset' 清空对话历史。")
    print("="*55)

    # 初始化 State
    state: AgentState = {
        "messages": [],
        "profile": CustomerProfile(),
        "has_proposed_solution": False,
        "dialog_status": "START",
        "last_intent": IntentType.GREETING
    }

    # 首先模拟 First Greeting (AI 主动破冰)
    print("\n[系统触发] AI 主动破冰...")
    res = first_greeting_node(state)
    for msg in res["messages"]:
        print(f"🤖 暴叔: {msg.content}")
        state["messages"].append(msg)

    while True:
        try:
            # 1. 获取用户输入
            user_input = input("\n👤 你: ").strip()
            if not user_input:
                continue
            
            # 在重定向到文件时，input的内容不会显示，所以这里显式打印出来
            print(f"👤 你: {user_input}")
            if user_input.lower() == 'exit':
                print("\n测试结束。")
                break
            if user_input.lower() == 'reset':
                state["messages"] = []
                print("\n--- 对话已重置 ---")
                continue

            # 2. 更新 State 并调用 chit_chat_node
            state["messages"].append(HumanMessage(content=user_input))
            state["last_intent"] = IntentType.CHIT_CHAT
            
            print("\n[AI 思考中...]")
            # 调用节点逻辑
            output = chit_chat_node(state)

            # 3. 展示结果
            if "messages" in output:
                for msg in output["messages"]:
                    print(f"🤖 暴叔: {msg.content}")
                    # 将 AI 的回复也存入历史，以便模拟连续对话
                    state["messages"].append(msg)
            
            # 如果有状态更新，同步到当前 state
            if "dialog_status" in output:
                state["dialog_status"] = output["dialog_status"]

        except KeyboardInterrupt:
            print("\n退出测试。")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    run_interactive_test()
