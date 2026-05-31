# ControlNet Style Transfer

[中文 README](README.md)

A lightweight image style transfer app built on `Stable Diffusion 1.5 + ControlNet (Canny)`. It keeps the structural layout of the source image while borrowing mood and color cues from a reference image, giving you a more controllable generation workflow.

`Python 3.10` or `3.11` is recommended. On the current machine, the app was verified to reach the model-download stage, but `3.13/3.14` should be treated as compatibility testing rather than the primary target environment.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-red)
![Diffusers](https://img.shields.io/badge/Diffusers-0.30.3-orange)
![Gradio](https://img.shields.io/badge/Gradio-4.44.1-green)
![Release](https://img.shields.io/badge/release-v0.1.0-blue)

## Features

- Uses ControlNet Canny to preserve structure and reduce uncontrolled generation drift.
- Combines source image, reference image, and prompt text for multi-factor control.
- Applies lightweight color matching so the output better follows the mood of the reference image.
- Loads models from local cache first, then falls back to online download when needed.
- Provides a simple Gradio UI for fast experimentation.
- Saves generated outputs automatically to `outputs/`.

## Project Structure

```text
ControlNet-Style-Transfer/
├─ app.py
├─ main.py
├─ LICENSE
├─ README.md
├─ README_EN.md
├─ requirements.txt
├─ CHANGELOG.md
├─ ROADMAP.md
├─ config/
│  └─ settings.py
├─ docs/
│  ├─ assets/
│  │  ├─ ui-preview.svg
│  │  └─ example-gallery.svg
│  └─ releases/
│     └─ v0.1.0.md
└─ src/
   ├─ __init__.py
   ├─ image_utils.py
   ├─ model_loader.py
   ├─ pipeline.py
   └─ ui.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-name/ControlNet-Style-Transfer.git
cd ControlNet-Style-Transfer
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

CPU:

```bash
pip install -r requirements.txt
```

NVIDIA CUDA:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

If you already installed `torch` and `torchvision` separately, you can also run:

```bash
pip install -r requirements.txt --no-deps
```

### 4. Launch the app

```bash
python main.py
```

Default URL:

```text
http://127.0.0.1:7860
```

On first launch, the app downloads:

- Base model: `SG161222/Realistic_Vision_V6.0_B1_noVAE`
- ControlNet model: `lllyasviel/sd-controlnet-canny`

## Usage

1. Upload a source image for structural guidance.
2. Upload a reference image for style and color direction.
3. Optionally add a prompt to reinforce details you want.
4. Tune `Strength / ControlNet Scale / CFG / Steps`.
5. Generate the result and review the output on the right.

## Screenshots / Example Images

The repository currently includes placeholder visual assets so the project page is presentation-ready. You can replace them later with real screenshots from your own generations.

### UI Preview

![UI Preview](docs/assets/ui-preview.svg)

### Example Flow

![Example Gallery](docs/assets/example-gallery.svg)

## Default Parameters

- `Strength`: How strongly the output departs from the source image.
- `ControlNet Scale`: How tightly the generated image follows source structure.
- `CFG`: Prompt guidance strength. Higher values may follow text better, but can become less natural.
- `Steps`: Inference steps. `20-40` is usually a good balance.

## Improvements in v0.1.0

- Normalizes input images to RGB, making grayscale and RGBA inputs safer.
- Prevents zero-dimension resize failures on extreme aspect ratios.
- Uses unique output filenames to reduce accidental overwrite.
- Exposes `ControlNet Scale / CFG / Steps` in the UI.
- Adds repo essentials such as `LICENSE`, install docs, release notes, roadmap, and preview assets.

## Known Limitations

- Results are still somewhat stochastic and prompt-sensitive.
- Style extraction is a lightweight heuristic, not a trained style encoder.
- First-time model download requires a stable network connection and enough disk space.
- There is no batch mode, queueing, or automated evaluation yet.

## Release Notes

- Current version: `v0.1.0`
- Full changelog: [CHANGELOG.md](CHANGELOG.md)
- Release notes: [docs/releases/v0.1.0.md](docs/releases/v0.1.0.md)
- Future plans: [ROADMAP.md](ROADMAP.md)

## License

This project is licensed under the [MIT License](LICENSE).
