# src/image_utils.py
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.exposure import match_histograms

from config.settings import (
    CANNY_HIGH_THRESHOLD,
    CANNY_LOW_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
)


def ensure_rgb(image):
    """Normalize arbitrary PIL-compatible input into an RGB image."""
    if image is None:
        return None

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def preprocess_image(image, target_size=DEFAULT_IMAGE_SIZE):
    """预处理图片：调整大小为 8 的倍数"""
    if image is None:
        return None

    image = ensure_rgb(image)
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # 确保是 8 的倍数（SD 要求）
    new_w = max(8, (new_w // 8) * 8)
    new_h = max(8, (new_h // 8) * 8)

    return image.resize((new_w, new_h), Image.LANCZOS)


def apply_color_match(source, reference):
    """强制将 source 的色调调整为 reference 的色调 (直方图匹配)"""
    if reference is None:
        return source
    src_arr = np.array(ensure_rgb(source))
    ref_arr = np.array(ensure_rgb(reference))
    
    # 匹配直方图
    try:
        matched = match_histograms(src_arr, ref_arr, channel_axis=-1)
        matched = np.clip(matched, 0, 255).astype("uint8")
        return Image.fromarray(matched)
    except Exception as e:
        print(f"⚠️ 色彩匹配失败: {e}")
        return source


def get_canny_image(image):
    """提取图片的 Canny 边缘图"""
    image = np.array(ensure_rgb(image))
    image = cv2.Canny(image, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)

    return Image.fromarray(image)


def extract_style_features(reference_image):
    """简单的风格描述提取 (辅助用)"""
    if reference_image is None:
        return ""
    img_array = np.array(ensure_rgb(reference_image))
    pixels = img_array.reshape(-1, 3)
    avg_color = np.mean(pixels, axis=0).astype(int)
    r, g, b = avg_color
    
    color_desc = ""
    if r > g and r > b:
        color_desc = "warm tones, reddish"
    elif g > r and g > b:
        color_desc = "cool tones, greenish"
    elif b > r and b > g:
        color_desc = "cool tones, bluish"
    else:
        color_desc = "neutral tones"
    
    brightness = np.mean(img_array)
    lighting_desc = "bright lighting" if brightness > 150 else "dark, moody lighting"
    
    return f"{color_desc}, {lighting_desc}"
