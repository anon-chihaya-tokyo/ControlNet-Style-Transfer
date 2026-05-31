# src/model_loader.py
import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, DDIMScheduler
from config.settings import DEVICE, BASE_MODEL_ID, CONTROLNET_ID

class ModelManager:
    def __init__(self):
        self.pipe = None
        self.controlnet = None

    def _load_smart(self, model_class, model_id, **kwargs):
        """智能加载：优先离线，失败则联网"""
        try:
            print(f"📂 [ModelManager] 尝试离线加载: {model_id} ...")
            return model_class.from_pretrained(model_id, local_files_only=True, **kwargs)
        except OSError as e:
            print(f"⚠️ 本地缓存不可用，切换联网下载: {model_id}")
            print(f"   详细原因: {e}")
            return model_class.from_pretrained(model_id, local_files_only=False, **kwargs)

    def load_models(self):
        print("🚀 正在初始化模型组件...")
        
        # 1. 加载 ControlNet
        self.controlnet = self._load_smart(
            ControlNetModel,
            CONTROLNET_ID,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )

        # 2. 加载主模型
        self.pipe = self._load_smart(
            StableDiffusionControlNetImg2ImgPipeline,
            BASE_MODEL_ID,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            safety_checker=None
        )

        # 3. 调度器与优化
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        if DEVICE == "cuda":
            print("⚡ 启用 CPU Offload 显存优化...")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        else:
            self.pipe.to(DEVICE)
        
        print("✅ 模型加载完毕！")
        return self.pipe
