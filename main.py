# main.py
import sys
import os

# 确保能导入 src 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model_loader import ModelManager
from src.ui import create_ui

if __name__ == "__main__":
    print("=========================================")
    print("   AI Style Transfer System (Pro Ver)    ")
    print("=========================================")
    
    # 1. 实例化管理器并加载模型
    manager = ModelManager()
    pipe = manager.load_models()
    
    # 2. 创建并启动 UI
    demo = create_ui(pipe)
    print("🌟 服务启动中，请访问下方链接...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)