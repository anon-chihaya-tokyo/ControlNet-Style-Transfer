import gradio as gr
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from PIL import Image
import numpy as np
import cv2
from skimage.exposure import match_histograms

print("🚀 正在初始化带 ControlNet 的风格迁移模型...")

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

def load_model_smart(model_class, model_id, **kwargs):
    """
    智能加载函数：
    1. 优先尝试 local_files_only=True (完全不联网，秒开)
    2. 如果本地没文件，再自动联网下载
    """
    try:
        print(f"📂 尝试离线加载本地缓存: {model_id} ...")
        # 核心修改：强制只看本地，不发任何网络请求
        return model_class.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception as e:
        print(f"⚠️ 本地未找到或损坏 ({str(e)})")
        print(f"🌐 正在尝试联网下载: {model_id} ...")
        # 只有本地失败了，才联网
        return model_class.from_pretrained(model_id, local_files_only=False, **kwargs)

# 1. 加载 ControlNet
# 注意：这里把原来的 ControlNetModel.from_pretrained 换成了我们的智能函数
controlnet = load_model_smart(
    ControlNetModel,
    "lllyasviel/sd-controlnet-canny",  # 或者是你改过的 softedge
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# 2. 加载 Stable Diffusion 主模型
# 确保这里是你选定的最新模型 ID
model_id = "SG161222/Realistic_Vision_V6.0_B1_noVAE"  # 或者是 "emilianJR/epiCRealism"

pipe = load_model_smart(
    StableDiffusionControlNetImg2ImgPipeline,
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None
)

# 使用 DDIM 采样器
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# 显存优化策略 (保留你之前的优化)
if device == "cuda":
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()

print("✅ 模型加载完成！")
# =============================================================================
# == 辅助函数 ==
# =============================================================================

def preprocess_image(image, target_size=512):
    """预处理图片：调整大小为 8 的倍数"""
    if image is None:
        return None
    
    # 保持宽高比缩放
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 确保是 8 的倍数（SD 要求）
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    return image.resize((new_w, new_h), Image.LANCZOS)

def apply_color_match(source, reference):
    """强制将 source 的色调调整为 reference 的色调"""
    src_arr = np.array(source)
    ref_arr = np.array(reference)
    # 匹配直方图
    matched = match_histograms(src_arr, ref_arr, channel_axis=-1)
    return Image.fromarray(matched.astype('uint8'))

def get_canny_image(image):
    """
    提取图片的 Canny 边缘图
    这是 ControlNet 的核心：告诉 AI 图片的线条在哪里
    """
    image = np.array(image)
    
    # Canny 边缘检测阈值
    low_threshold = 100
    high_threshold = 200
    
    image = cv2.Canny(image, low_threshold, high_threshold)
    
    # 将单通道边缘图转换为三通道 (RGB)，因为 ControlNet 需要 RGB 输入
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    
    return Image.fromarray(image)


def extract_style_features(reference_image):
    """从参考图提取简单的风格描述 (保留原逻辑作为辅助)"""
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


# =============================================================================
# == 核心生成逻辑 ==
# =============================================================================

@torch.no_grad()
def style_transfer(source_image, reference_image, style_strength=0.75, 
                   custom_prompt="", seed=42):
    """
    带 ControlNet 的风格迁移函数
    """
    
    if source_image is None:
        raise gr.Error("请上传源图片！")
    
    # 1. 预处理源图
    source_image = preprocess_image(source_image)
    
    # 2. 制作 ControlNet 需要的边缘控制图
    canny_image = get_canny_image(source_image)
    
    # 3. 构建提示词
    # 提取参考图特征（可选，如果不想用自动提取，可以留空）
    style_desc = extract_style_features(reference_image) if reference_image else ""
    
    # 基础高质量词 + 用户输入 + 自动提取的风格
    base_prompt = "masterpiece, best quality, high resolution"
    
    if custom_prompt:
        prompt = f"{base_prompt}, cinematic lighting, detailed texture, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain,{custom_prompt}, {style_desc}"
    else:
        # 默认提示词，强调风格化
        prompt = f"{base_prompt}, cinematic lighting, detailed texture, RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain,{style_desc}"
    
    negative_prompt = f"nsfw, nude, naked, cleavage, nipples, revealing clothes, lingerie, bikini, "  # 核心防
    "bad anatomy, bad hands, missing fingers, extra fingers, three hands, "        # 防肢体崩坏
    "deformed, blurry, low quality, jpeg artifacts, text, watermark, signature, " # 防画质差
    "makeup, plastic skin, doll, 3d render, cartoon"
    
    print(f"生成提示词: {prompt}")
    
    # 4. 设置随机种子
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # 5. 生成 (Img2Img + ControlNet)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=source_image,           # 原图 (用于 img2img 颜色参考)
        control_image=canny_image,    # 控制图 (用于 ControlNet 锁定结构)
        
        # 关键参数调整
        strength=style_strength,             # 风格化强度 (0.6-0.9 均可，因为有 ControlNet 锁结构)
        controlnet_conditioning_scale=0.5,   # ControlNet 权重 (1.0 = 严格遵守线条)
        guidance_scale=7.5,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    result = apply_color_match(result, reference_image)
    return result, canny_image  # 返回结果和边缘图（方便调试）


# =============================================================================
# == Gradio 界面 ==
# =============================================================================

with gr.Blocks(title="AI 风格迁移 (ControlNet版)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 升级版风格迁移系统 (Powered by ControlNet)
    
    **升级说明：** 引入了 ControlNet (Canny) 技术。现在你可以放心调高"风格强度"，系统会严格锁定源图片的线条结构，不再担心脸崩或眼镜消失！
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 输入")
            source_img = gr.Image(type="pil", label="源图片（保留结构）")
            reference_img = gr.Image(type="pil", label="参考图片（提供风格/颜色参考）")
            
            with gr.Accordion("🔧 参数设置", open=True):
                style_strength = gr.Slider(
                    0.3, 1.0, value=0.75, step=0.05,
                    label="风格强度 (建议 0.6 - 0.9)"
                )
                custom_prompt = gr.Textbox(
                    label="风格描述 (强烈建议手动输入)",
                    placeholder="例如: oil painting style, van gogh, blue swirling sky"
                )
                seed = gr.Slider(
                    0, 999999, value=42, step=1,
                    label="随机种子"
                )
            
            generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### ✨ 输出")
            output_img = gr.Image(type="pil", label="风格迁移结果")
            
            with gr.Accordion("👀 查看结构控制图 (调试用)", open=False):
                canny_debug_img = gr.Image(type="pil", label="系统提取的边缘图")

    # 绑定按钮
    generate_btn.click(
        fn=style_transfer,
        inputs=[source_img, reference_img, style_strength, custom_prompt, seed],
        outputs=[output_img, canny_debug_img]
    )

print("🌟 启动 Gradio 界面...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)