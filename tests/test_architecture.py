import unittest
import sys
import os
import time
from pprint import pprint

# ==========================================
# 1. 路径修复 (确保能找到根目录的模块)
# ==========================================
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设当前脚本在 tests/ 下，或者就在根目录下)
if os.path.basename(current_dir) == 'tests':
    project_root = os.path.dirname(current_dir)
else:
    project_root = current_dir
sys.path.append(project_root)

# ==========================================
# 2. 导入模块
# ==========================================
from langchain_core.messages import HumanMessage, AIMessage
from state import AgentState, CustomerProfile, BudgetInfo, BudgetPeriod, IntentType
# 导入待测路由函数
from router import core_router

# 导入节点 (用于集成测试)
from nodes.perception import classifier_node, extractor_node
from agent_graph import app as graph_app

# 颜色打印工具
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def print_test_header(name):
    print(f"\n{Colors.HEADER}=================================================={Colors.ENDC}")
    print(f"{Colors.HEADER}🧪 测试场景: {name}{Colors.ENDC}")
    print(f"{Colors.HEADER}=================================================={Colors.ENDC}")

class TestBaoshuArchitecture(unittest.TestCase):

    # =================================================================
    # 1. 🧠 决策层测试 (Router Logic) - 纯逻辑验证
    # =================================================================
    
    def test_01_router_art_director(self):
        """测试：艺术生通道优先级 (应高于低预算和资料不全)"""
        print_test_header("Router: 艺术生通道")
        
        # 模拟 State: 意图是艺术，画像不全也不管
        mock_state = {
            "last_intent": IntentType.ART_CONSULTING,
            "profile": CustomerProfile(), # 空画像
            "dialog_status": "START"
        }
        
        next_node = core_router(mock_state)
        print(f"输入意图: ART_CONSULTING -> 路由结果: {next_node}")
        
        self.assertEqual(next_node, "art_director", "❌ 艺术生应该直通 art_director！")
        print(f"{Colors.OKGREEN}✅ 通过{Colors.ENDC}")

    def test_02_router_low_budget_priority(self):
        """🔥 核心测试：低预算 > 资料不全 (资料不全也要进低预算)"""
        print_test_header("Router: 低预算抢跑逻辑")
        
        # 构造一个“资料严重缺失”但“预算很低”的画像
        profile = CustomerProfile()
        profile.budget.amount = 5 # 5万
        # profile.educationStage 是 None (缺失)
        # profile.destination_preference 是 None (缺失)
        
        self.assertFalse(profile.is_complete, "测试前提：画像必须是不完整的")
        
        mock_state = {
            "last_intent": IntentType.NEED_CONSULTING,
            "profile": profile,
            "dialog_status": "CONSULTING"
        }
        
        next_node = core_router(mock_state)
        print(f"输入: 预算5万 + 资料缺失 -> 路由结果: {next_node}")
        
        # 旧逻辑是 interviewer，新逻辑必须是 low_budget
        self.assertEqual(next_node, "low_budget", "❌ 低预算客户被 Interviewer 拦截了！新逻辑应直通 low_budget！")
        print(f"{Colors.OKGREEN}✅ 通过{Colors.ENDC}")

    def test_03_router_low_budget_explicit_flag(self):
        """测试：Intent 标记为 LOW_BUDGET (如负债)"""
        print_test_header("Router: 负债/显性低预算")
        
        mock_state = {
            "last_intent": IntentType.LOW_BUDGET,  # 🔥 直接通过意图标记
            "profile": CustomerProfile(), # 资料全空
            "dialog_status": "CONSULTING"
        }
        
        next_node = core_router(mock_state)
        print(f"输入: intent=LOW_BUDGET -> 路由结果: {next_node}")
        
        self.assertEqual(next_node, "low_budget", "❌ 低预算意图路由失效！")
        print(f"{Colors.OKGREEN}✅ 通过{Colors.ENDC}")

    def test_04_router_interviewer_blocker_normal_user(self):
        """测试：普通用户(非低预算) 资料不全时，必须被 Interviewer 拦截"""
        print_test_header("Router: 普通用户资料补全")
        
        # 普通用户：预算未知 (-1) 或 预算正常 (50)
        profile = CustomerProfile()
        profile.budget.amount = -1 # 未知预算
        
        mock_state = {
            "last_intent": IntentType.NEED_CONSULTING,
            "profile": profile, 
            "dialog_status": "START",
            "is_low_budget": False
        }
        
        next_node = core_router(mock_state)
        print(f"输入: 普通用户 + 资料缺失 -> 路由结果: {next_node}")
        
        self.assertEqual(next_node, "interviewer", "❌ 普通用户资料不全应该去 interviewer！")
        print(f"{Colors.OKGREEN}✅ 通过{Colors.ENDC}")

    def test_05_router_consultant_success(self):
        """测试：普通用户，资料齐全，进入 Consultant"""
        print_test_header("Router: 常规 Consultant")
        
        # 构造完美画像
        profile = CustomerProfile()
        profile.user_role = "学生"
        profile.educationStage = "本科"
        profile.budget.amount = 50
        profile.budget.period = BudgetPeriod.ANNUAL
        profile.destination_preference = "境外方向"
        profile.academic_background = "GPA 3.0"
        
        self.assertTrue(profile.is_complete, "测试前提：画像必须是完整的")
        
        mock_state = {
            "last_intent": IntentType.NEED_CONSULTING,
            "profile": profile,
            "dialog_status": "PROFILING"
        }
        
        next_node = core_router(mock_state)
        print(f"输入: 资料齐全 + 预算正常 -> 路由结果: {next_node}")
        
        self.assertEqual(next_node, "consultant", "❌ 资料齐全应该去 consultant！")
        print(f"{Colors.OKGREEN}✅ 通过{Colors.ENDC}")

    # =================================================================
    # 2. 👁️ 感知层测试 (Perception Nodes) - 实战测试
    # =================================================================

    def test_06_classifier_real_llm(self):
        """测试：Classifier 节点 LLM 调用"""
        print_test_header("Node: Classifier 实战 (检测负债)")

        # 测试负债关键词能否触发 LOW_BUDGET 意图
        msg_debt = [HumanMessage(content="我欠了80w房贷，能跑路吗？")]
        state_debt = {"messages": msg_debt, "profile": CustomerProfile(), "last_intent": None}

        try:
            res_debt = classifier_node(state_debt)
            print(f"用户: {msg_debt[0].content}")
            print(f"分类结果: {res_debt['last_intent']}")

            # 新架构：通过 intent 判断是否低预算，不再使用 is_low_budget 标记
            self.assertEqual(res_debt['last_intent'], IntentType.LOW_BUDGET, "❌ 未识别出负债导致的低预算意图")
            print(f"{Colors.OKGREEN}✅ Classifier 正常工作{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}❌ LLM 调用失败: {e}{Colors.ENDC}")

    # =================================================================
    # 3. 💪 Graph 集成测试
    # =================================================================
    
    def test_07_graph_integration_low_budget(self):
        """测试：Graph 完整流程 (低预算抢跑测试)"""
        print_test_header("Graph: 低预算抢跑集成测试")
        
        # 模拟用户直接说也没钱
        print(f"{Colors.OKBLUE}--- 用户进线: '我有5万预算' ---{Colors.ENDC}")
        inputs = {"messages": [HumanMessage(content="你好暴叔，我只有5万预算，想出国")]}
        config = {"configurable": {"thread_id": "test_thread_low_budget"}}
        
        try:
            # 运行 Graph
            events = list(graph_app.stream(inputs, config=config))
            node_names = [list(e.keys())[0] for e in events]
            
            print(f"执行路径: {node_names}")
            
            # 验证路径
            # 1. 应该经过 classifier 和 extractor (并行)
            self.assertIn("classifier", node_names)
            self.assertIn("extractor", node_names)
            
            # 2. 关键验证：应该跳过 interviewer，直接进 low_budget
            # (虽然此时资料肯定不全，只有预算)
            if "interviewer" in node_names:
                print(f"{Colors.FAIL}❌ 失败：还是进了 Interviewer 查户口！{Colors.ENDC}")
                self.fail("Low Budget flow failed")
            
            self.assertIn("low_budget", node_names, "❌ 没进 low_budget 节点！")
            
            # 打印最后回复
            final_event = events[-1]
            last_node = list(final_event.keys())[0]
            ai_msg = final_event[last_node]['messages'][0].content
            print(f"AI ({last_node}): {ai_msg[:50]}...")
            
            print(f"{Colors.OKGREEN}✅ Graph 低预算路径测试通过{Colors.ENDC}")
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Graph 运行失败: {e}{Colors.ENDC}")

if __name__ == '__main__':
    unittest.main()