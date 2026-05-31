# Changelog

## v0.1.0 - 2026-05-31

Initial public baseline release.

### Added

- MIT `LICENSE`
- Full README with installation and usage instructions
- `ROADMAP.md`
- Repository preview assets under `docs/assets/`
- `app.py` entry point for app-style launch
- Adjustable `ControlNet Scale / CFG / Steps` controls in the Gradio UI

### Changed

- Pinned dependency versions in `requirements.txt`
- Refactored `main.py` to expose `build_demo()` and `main()`
- Centralized more defaults in `config/settings.py`
- Improved output naming to avoid accidental overwrite

### Fixed

- RGB normalization for grayscale and RGBA images
- Safer resize logic for extreme aspect ratios
- More explicit offline-to-online fallback handling when loading models
