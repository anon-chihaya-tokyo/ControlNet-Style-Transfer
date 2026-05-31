# src/ui.py
import gradio as gr
from config.settings import (
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_STYLE_STRENGTH,
)
from src.pipeline import run_style_transfer

def create_ui(pipe):
    """构建 Gradio 界面，传入已加载的 pipe"""
    
    def inference_wrapper(source, ref, strength, prompt, seed, controlnet_scale, guidance_scale, steps):
        # 包装函数，将 pipe 注入到业务逻辑中
        return run_style_transfer(
            pipe,
            source,
            ref,
            strength,
            prompt,
            seed,
            controlnet_scale,
            guidance_scale,
            steps,
        )

    with gr.Blocks(title="AI 风格迁移 Pro", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 AI 风格迁移系统 (工程版)
        **Core**: Stable Diffusion 1.5 + ControlNet (Canny) | **Model**: Realistic Vision V6.0
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 输入设置")
                source_img = gr.Image(type="pil", label="源图片 (结构参考)")
                reference_img = gr.Image(type="pil", label="参考图片 (风格/色彩)")
                
                with gr.Accordion("🔧 高级参数", open=True):
                    style_strength = gr.Slider(0.3, 1.0, value=DEFAULT_STYLE_STRENGTH, step=0.05, label="风格化强度 (Strength)")
                    custom_prompt = gr.Textbox(label="补充描述 (Prompt)", placeholder="例如: 1girl, wearing glasses, black overalls")
                    seed = gr.Slider(0, 999999, value=DEFAULT_SEED, step=1, label="随机种子 (Seed)")
                    controlnet_scale = gr.Slider(0.1, 1.5, value=DEFAULT_CONTROLNET_SCALE, step=0.05, label="结构控制权重 (ControlNet Scale)")
                    guidance_scale = gr.Slider(1.0, 15.0, value=DEFAULT_GUIDANCE_SCALE, step=0.5, label="提示词引导强度 (CFG)")
                    steps = gr.Slider(10, 60, value=DEFAULT_NUM_INFERENCE_STEPS, step=1, label="推理步数 (Steps)")
                
                btn = gr.Button("🚀 生成作品", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### ✨ 结果预览")
                output_img = gr.Image(type="pil", label="最终结果")
                with gr.Accordion("👀 调试视图", open=False):
                    debug_img = gr.Image(type="pil", label="Canny 边缘提取图")
        
        btn.click(
            fn=inference_wrapper,
            inputs=[source_img, reference_img, style_strength, custom_prompt, seed, controlnet_scale, guidance_scale, steps],
            outputs=[output_img, debug_img]
        )
        
    return demo
