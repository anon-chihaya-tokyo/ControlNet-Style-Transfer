# src/pipeline.py
import os
import time
import uuid

import gradio as gr
import torch

from config.settings import (
    DEFAULT_NEG_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_POS_PROMPT,
    DEFAULT_CONTROLNET_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEVICE,
    OUTPUT_DIR,
)
from src.image_utils import (
    apply_color_match,
    extract_style_features,
    get_canny_image,
    preprocess_image,
)


def build_prompt(custom_prompt, style_desc):
    prompt_parts = [DEFAULT_POS_PROMPT]
    if custom_prompt and custom_prompt.strip():
        prompt_parts.append(custom_prompt.strip())
    if style_desc:
        prompt_parts.append(style_desc)
    return ", ".join(prompt_parts)


def run_style_transfer(
    pipe,
    source_image,
    reference_image,
    style_strength,
    custom_prompt,
    seed,
    controlnet_scale=DEFAULT_CONTROLNET_SCALE,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
):
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
    full_prompt = build_prompt(custom_prompt, style_desc)
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
        controlnet_conditioning_scale=controlnet_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        generator=generator
    ).images[0]
    
    # 5. 后处理：色彩匹配
    if reference_image:
        result = apply_color_match(result, reference_image)
    
    # 6. 自动保存结果 (新增功能)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    save_path = os.path.join(OUTPUT_DIR, f"result_{timestamp}_{unique_id}.png")
    result.save(save_path)
    print(f"💾 结果已保存至: {save_path}")
    
    return result, canny_image
