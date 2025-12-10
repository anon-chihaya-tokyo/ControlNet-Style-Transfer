# 🎨 基于 Stable Diffusion 的图像风格迁移系统

## 项目简介

本项目实现了基于深度学习的图像风格迁移功能，使用预训练的 Stable Diffusion 模型，能够将参考图片的风格（色调、光照、纹理）迁移到源图片上，同时保持源图片的内容结构。

## 技术栈

- **深度学习框架**: PyTorch
- **预训练模型**: Stable Diffusion v1.5
- **推理库**: Diffusers (Hugging Face)
- **用户界面**: Gradio
- **图像处理**: OpenCV, Pillow

## 核心功能

1. **自动风格提取**: 从参考图自动分析颜色、光照特征
2. **结构保持**: 使用 img2img 管道保留源图内容
3. **可调参数**: 风格强度、随机种子、自定义提示词
4. **实时预览**: Gradio 界面支持即时查看结果

## 安装与运行

### 环境要求

- Python 3.8+
- CUDA 11.8+ (GPU 推荐，CPU 也可运行但较慢)
- 至少 8GB 显存 (GPU) 或 16GB 内存 (CPU)

### 安装步骤
```bash
# 1. 创建 conda 环境
conda create -n style_transfer python=3.10 -y
conda activate style_transfer

# 2. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 运行
bash run.sh
# 或直接运行
python app.py
```

### 首次运行

首次运行时，系统会自动从 Hugging Face 下载 Stable Diffusion 模型（约 4GB），请耐心等待。

## 使用方法

1. 启动后访问 `http://localhost:7860`
2. 上传源图片（想要保留内容的图片）
3. 上传参考图片（想要提取风格的图片）
4. 调整风格强度（推荐 0.7-0.85）
5. 点击"生成风格迁移"按钮
6. 等待 10-30 秒即可看到结果

## 项目结构
```
style_transfer_project/
├── app.py              # 主程序
├── requirements.txt    # 依赖列表
├── run.sh             # 启动脚本
├── README.md          # 项目说明
└── examples/          # 示例图片（可选）
```

## 技术原理

### 1. 模型选择
使用 Stable Diffusion v1.5 作为基础模型，该模型在大规模图像数据集上预训练，具有强大的图像生成和编辑能力。

### 2. 风格迁移流程
1. **特征提取**: 分析参考图的颜色分布和亮度
2. **提示词生成**: 将视觉特征转换为文本描述
3. **条件生成**: 使用 img2img 管道，以源图为初始输入
4. **迭代优化**: 通过 30 步 DDIM 采样生成结果

### 3. 关键参数
- `strength`: 控制风格迁移强度（0.3-1.0）
- `guidance_scale`: 控制提示词引导强度（默认 7.5）
- `num_inference_steps`: 生成步数（默认 30）

## 性能优化

- 使用 `torch.float16` 混合精度（GPU）
- 启用注意力切片（`enable_attention_slicing`）节省显存
- 图像预处理确保尺寸为 8 的倍数

## 注意事项

- 建议源图和参考图的主体内容相似（如都是人脸）
- 风格强度过高可能导致源图内容失真
- CPU 模式下生成速度较慢（约 2-5 分钟/张）

