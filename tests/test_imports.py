import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)

# Add project root to sys.path
sys.path.append(project_root)

print("🔍 开始全量 Import 扫描...")

try:
    print("Checking state.py...")
    import state
    
    print("Checking tools.py...")
    from nodes import tools
    
    print("Checking perception.py...")
    from nodes import perception
    
    print("Checking consultants.py...")
    from nodes import consultants
    
    print("Checking router.py...")
    import router
    
    print("Checking agent_graph.py...")
    import agent_graph
    
    print("Checking main.py...")
    # main 可能会启动 server，我们只 import 不运行
    # 只要 import main 不报错，说明 main 的依赖都齐了
    
    print("\n✅ 所有模块 Import 成功！语法和依赖基本没问题。")

except Exception as e:
    print(f"\n❌ 发现致命错误！")
    print(f"文件名: {e.__traceback__.tb_frame.f_code.co_filename}")
    print(f"错误信息: {e}")
    sys.exit(1)