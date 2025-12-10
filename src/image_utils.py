# src/image_utils.py
import cv2
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms

def preprocess_image(image, target_size=512):
    """预处理图片：调整大小为 8 的倍数"""
    if image is None:
        return None
    
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 确保是 8 的倍数（SD 要求）
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    return image.resize((new_w, new_h), Image.LANCZOS)

def apply_color_match(source, reference):
    """强制将 source 的色调调整为 reference 的色调 (直方图匹配)"""
    if reference is None:
        return source
    src_arr = np.array(source)
    ref_arr = np.array(reference)
    
    # 匹配直方图
    try:
        matched = match_histograms(src_arr, ref_arr, channel_axis=-1)
        return Image.fromarray(matched.astype('uint8'))
    except Exception as e:
        print(f"⚠️ 色彩匹配失败: {e}")
        return source

def get_canny_image(image):
    """提取图片的 Canny 边缘图"""
    image = np.array(image)
    
    low_threshold = 100
    high_threshold = 200
    
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    
    return Image.fromarray(image)

def extract_style_features(reference_image):
    """简单的风格描述提取 (辅助用)"""
    if reference_image is None:
        return ""
    img_array = np.array(reference_image)
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