from src.model_loader import ModelManager
from src.ui import create_ui

def build_demo():
    manager = ModelManager()
    pipe = manager.load_models()
    return create_ui(pipe)


def main():
    print("=========================================")
    print("   AI Style Transfer System (Pro Ver)    ")
    print("=========================================")

    demo = build_demo()
    print("🌟 服务启动中，请访问下方链接...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
