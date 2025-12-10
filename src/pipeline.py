# src/pipeline.py
import torch
import os
import time
from config.settings import DEVICE, DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT, OUTPUT_DIR
from src.image_utils import preprocess_image, get_canny_image, extract_style_features, apply_color_match
import gradio as gr

def run_style_transfer(pipe, source_image, reference_image, style_strength, custom_prompt, seed):
    """
    核心生成函数
    参数:
        pipe: 已加载的模型管道
        source_image: 原图
        reference_image: 参考图
        style_strength: 风格强度
        custom_prompt: 用户输入的提示词
        seed: 随机种子
    """
    if source_image is None:
        raise gr.Error("请上传源图片！")
    
    # 1. 预处理
    source_image = preprocess_image(source_image)
    canny_image = get_canny_image(source_image)
    
    # 2. 构建提示词
    style_desc = extract_style_features(reference_image) if reference_image else ""
    full_prompt = f"{DEFAULT_POS_PROMPT}, {custom_prompt}, {style_desc}"
    print(f"🎨 生成提示词: {full_prompt}")
    
    # 3. 设置种子
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    
    # 4. 推理生成
    result = pipe(
        prompt=full_prompt,
        negative_prompt=DEFAULT_NEG_PROMPT,
        image=source_image,           # Img2Img 输入
        control_image=canny_image,    # ControlNet 输入
        strength=style_strength,
        controlnet_conditioning_scale=0.5, # 推荐权重
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    # 5. 后处理：色彩匹配
    if reference_image:
        result = apply_color_match(result, reference_image)
    
    # 6. 自动保存结果 (新增功能)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}.png")
    result.save(save_path)
    print(f"💾 结果已保存至: {save_path}")
    
    return result, canny_image