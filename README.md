# 🎨 Enterprise-Grade AI Style Transfer System
> 基于 Stable Diffusion 1.5 + ControlNet 的高保真图像风格迁移系统（工程重构版）

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red) ![Diffusers](https://img.shields.io/badge/Diffusers-0.21%2B-orange) ![Gradio](https://img.shields.io/badge/Gradio-WebUI-green)

## 📖 项目简介

本项目解决了传统 AI 绘画“抽卡”不可控的问题，通过引入 **ControlNet (Canny)** 技术，实现了对图像结构（如脸型、眼镜、姿态）的严格锁定，同时利用 **Realistic Vision V6.0** 模型提供SOTA级别的写实光影效果。

**核心价值**：Structure (结构) + Style (风格) 的完美解耦与重组。

## ✨ 核心特性 (Key Features)

- **🏛️ 模块化架构设计**：采用分层架构（配置层、核心层、表现层），高内聚低耦合，易于维护和扩展。
- **🧠 智能离线优先加载**：内置 `ModelManager`，启动时优先读取本地缓存。即使断网也能秒级启动，彻底告别 HuggingFace 连接超时烦恼。
- **⚡ 极致显存优化**：集成 `CPU Offload` 与 `VAE Slicing` 技术，支持在 **4GB 显存** 的笔记本（如 GTX 1650/RTX 3050）上流畅运行大模型。
- **🎨 自动色彩校正**：引入直方图匹配算法（Histogram Matching），强制生成图继承参考图的色调分布。
- **🛡️ 企业级安全过滤**：内置强力 Negative Prompt 策略，有效过滤 NSFW 内容及肢体崩坏。
- **💾 结果自动归档**：所有生成结果自动按时间戳保存至 `outputs/` 目录，不再丢失灵感。

## 📂 项目架构

本项目遵循工业级 Python 工程规范：

```text
style_transfer_project/
├── config/                 # ⚙️ 配置中心
│   └── settings.py         # 模型ID、默认Prompt、路径配置（在此修改模型）
├── src/                    # 🧠 核心源码
│   ├── model_loader.py     # 模型生命周期管理（单例模式/离线加载/显存优化）
│   ├── pipeline.py         # 业务管线（生成逻辑/色彩校正/自动保存）
│   ├── image_utils.py      # 图像处理算法（Canny边缘提取/直方图匹配）
│   └── ui.py               # 前端界面（Gradio 依赖注入设计）
├── outputs/                # 💾 结果产出（自动忽略上传）
├── main.py                 # 🚀 系统启动入口
└── requirements.txt        # 📦 依赖清单
