# config/settings.py
import torch
import os

# == 路径配置 ==
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# == 设备配置 ==
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# == 模型配置 ==
# 基础模型：Realistic Vision V6.0 B1 (2025年版本)
BASE_MODEL_ID = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
# ControlNet 模型：Canny 边缘检测
CONTROLNET_ID = "lllyasviel/sd-controlnet-canny"

# == 默认提示词配置 ==
DEFAULT_POS_PROMPT = (
    "masterpiece, best quality, high resolution, "
    "cinematic lighting, detailed texture, RAW photo, "
    "subject, 8k uhd, dslr, soft lighting, film grain"
)

# 强力防崩坏负面提示词
DEFAULT_NEG_PROMPT = (
    "nsfw, nude, naked, cleavage, nipples, revealing clothes, lingerie, bikini, "
    "bad anatomy, bad hands, missing fingers, extra fingers, three hands, "
    "deformed, blurry, low quality, jpeg artifacts, text, watermark, signature, "
    "makeup, plastic skin, doll, 3d render, cartoon"
)