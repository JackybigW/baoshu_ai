# tests/test_architecture.py
import sys
import os
import unittest

# ==========================================
# 1. 强制修复路径 (Magic Fix)
# ==========================================
# 获取当前文件 (test_architecture.py) 的目录 -> baoshu_ai/test
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 -> baoshu_ai
project_root = os.path.dirname(current_dir)

# 核心：把根目录加到 Python 的搜索路径最前面！
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"🚀 正在运行架构测试...")
print(f"📂 项目根目录已挂载: {project_root}")

# ==========================================
# 2. 测试类
# ==========================================
class TestBaoshuAgent(unittest.TestCase):
    
    def test_01_imports_and_syntax(self):
        """测试核心模块导入"""
        try:
            # ✅ 修复：现在 tools 在 nodes 下面，必须带 nodes 前缀
            import state
            import agent_graph
            from nodes import consultants, router, tools 
            print("✅ [01] 核心模块 Import 测试通过")
        except ImportError as e:
            self.fail(f"❌ 导入失败: {e}\n👉 请检查：tools.py 是否在 nodes/ 文件夹里？")
        except SyntaxError as e:
            self.fail(f"❌ 语法错误: {e}")

    def test_02_tools_definition(self):
        """测试工具定义"""
        # ✅ 修复：引用路径改为 nodes.tools
        try:
            from nodes.tools import search_products, summon_specialist_tool
            self.assertTrue(callable(search_products), "❌ search_products 不是函数")
            self.assertTrue(hasattr(summon_specialist_tool, "name"), "❌ summon_specialist_tool 未装饰")
            print("✅ [02] 工具函数检查通过")
        except ImportError:
             self.fail("❌ 找不到 nodes.tools，请确认 tools.py 已移动到 nodes 文件夹！")

    def test_03_state_model(self):
        """测试状态模型"""
        from state import CustomerProfile
        profile = CustomerProfile()
        self.assertEqual(profile.budget.amount, -1)
        print("✅ [03] State 模型检查通过")

    def test_04_nodes_structure(self):
        """测试节点函数"""
        # ✅ 修复：引用路径正确
        from nodes.consultants import consultant_node, sales_node
        self.assertTrue(callable(consultant_node))
        print("✅ [04] 业务节点检查通过")

    def test_05_graph_compilation(self):
        """测试图编译"""
        try:
            from agent_graph import app
            self.assertIsNotNone(app)
            print("✅ [05] LangGraph 编译测试通过")
        except Exception as e:
            self.fail(f"❌ 图编译失败: {e}")

    def test_06_excel_loading(self):
        """测试 Excel"""
        import pandas as pd
        # ✅ 修复：确保读取路径是相对于项目根目录的
        excel_path = os.path.join(project_root, "products_intro.xlsx")
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                self.assertFalse(df.empty)
                print("✅ [06] Excel 读取测试通过")
            except:
                self.fail("❌ Excel 文件损坏")
        else:
            print(f"⚠️ [06] 跳过: 未找到 Excel ({excel_path})")

if __name__ == '__main__':
    unittest.main(verbosity=2)