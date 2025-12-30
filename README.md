# User Demo: Multimodal AI Pipeline for Synthetic Data Generation

An end-to-end pipeline that combines Large Language Models (LLMs), Diffusion Models, and Vision-Language Models to generate synthetic images with automatic object detection annotations. This pipeline is designed for creating high-quality training datasets for computer vision and robotics applications.

## Pipeline Overview

![End to End Flow](asset/End%20to%20End%20Flow.png)

## Getting Started

This repository contains an end-to-end multimodal AI pipeline for synthetic data generation and automatic annotation.  
Follow the **Installation** and **Usage** sections below to set up and run the pipeline locally.

## Description

This project implements a complete multimodal AI workflow that transforms simple text prompts into annotated synthetic images. The pipeline consists of five integrated stages:

1. **Prompt Enhancement** - Uses LLaMA2 via Ollama to enrich user prompts with detailed visual attributes, lighting conditions, and scene composition while respecting token limits (512 for Flux, 72 for Stable Diffusion)

2. **Image Generation** - Generates high-quality images using state-of-the-art diffusion models (Flux or Stable Diffusion) with configurable aspect ratios and batch generation support

3. **Interactive Preview** - Allows users to review generated images and download them in PNG or JPEG formats

4. **Automatic Annotation** - Employs Grounding DINO for precise object detection and bounding box generation, supporting 16 predefined object classes

5. **Export & Verification** - Outputs YOLO-format annotations with confidence scores and class labels, ready for training computer vision models

The pipeline is specifically tailored for industrial and robotics scenarios, with support for objects commonly found in warehouse and manufacturing environments.

## Features

### Core Capabilities
- **Token-Aware Prompt Enhancement**: Automatically optimizes prompts to fit model-specific token limits while maximizing detail
- **Multi-Model Support**: Choose between Flux.1-dev (high quality) and Stable Diffusion (fast generation)
- **Flexible Aspect Ratios**: Generate images in 1:1, 16:9, 4:3, or 2:1 ratios
- **Batch Generation**: Create multiple variations from a single prompt
- **Grounding DINO Integration**: State-of-the-art open-vocabulary object detection
- **YOLO Format Export**: Annotations ready for training YOLOv12 and similar models
- **Interactive Web Interface**: User-friendly Streamlit application with visual progress tracking

### Supported Object Classes
The pipeline currently supports detection of 16 object classes:
```
traffic_cone, fire_extinguisher, cardbox, pallet, bottle, vase, chair, monitor, 
plastic_box, shoe, apple, robotic_arm, cup, screw_driver, plier, hammer
```


## Installation

### Prerequisites

**System Requirements:**
- Python 3.10 or higher
- CUDA-compatible GPU (recommended: 8GB+ VRAM for Flux, 4GB+ for Stable Diffusion)
- 20GB+ free disk space for models
- Ubuntu 20.04+ or Windows 10+ (tested on Ubuntu 22.04)

**Required External Services:**
- [Ollama](https://ollama.ai/) - Local LLM inference server
- [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) - Vision-language annotation framework

### Step-by-Step Installation

1. **Clone the repository:**
```bash
git clone https://github.com/monikaits44/synthetic-data-generation-annotation.git
cd synthetic-data-generation-annotation
```

2. **Create and activate a virtual environment:**
```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Ollama and pull LLaMA2:**
```bash
# Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama2:latest

# Windows: Download from https://ollama.ai/download
# Then run: ollama pull llama2:latest
```

5. **Set up Grounded-SAM-2:**
```bash
# Clone the repository
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2

# Install dependencies
pip install -e .

# Download checkpoints
cd checkpoints
bash download_ckpts.sh

# Return to project root
cd ../..
```

6. **Configure model paths:**

Edit the configuration in each module file or set environment variables:

```bash
# For Flux model
export FLUX_PATH="/path/to/flux/model"

# For Stable Diffusion
export STABLE_DIFFUSION_PATH="/path/to/stable-diffusion"

# For Grounding DINO
export GROUNDING_DINO_PATH="/path/to/Grounded-SAM-2"
```

7. **Verify installation:**
```bash
python verify_pipeline.py
```

Expected output: All tests should pass with ✓ symbols indicating successful component initialization.

## Usage

### Starting the Application

1. **Ensure Ollama is running:**
```bash
# Check if Ollama is active
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

2. **Launch the Streamlit application:**
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Basic Workflow Example

**Stage 1: Input Configuration**
```
Prompt: "A robotic arm picking up a cardboard box in an industrial warehouse"
Model: Flux
```

**Stage 2: Prompt Enhancement**
The LLM enhances the prompt with:
- Lighting conditions (bright overhead lighting, LED panels)
- Visual details (yellow robotic arm, brown cardboard, metallic gripper)
- Scene composition (concrete floor, metal shelving in background)
- Camera perspective (eye-level, slight angle)

**Stage 3: Image Generation**
```
Prompt Type: Enhanced
Aspect Ratio: 16:9 (1024×576)
Number of Images: 3
Format: PNG
```

**Stage 4: Preview & Download**
- Review generated images side-by-side
- Download individual images or all at once

**Stage 5: Annotation**
```
Annotation Mode: Grounding DINO
Detection Prompt: "cardbox, robotic arm"
```

Output files saved to:
```
pipeline_outputs/
├── generated_images/
│   ├── flux_20241221_143022_1.png
│   ├── flux_20241221_143022_2.png
│   └── flux_20241221_143022_3.png
└── annotations/
    ├── flux_20241221_143022_1_annotated.png
    ├── flux_20241221_143022_1.txt
    ├── flux_20241221_143022_2_annotated.png
    └── flux_20241221_143022_2.txt
```

### Annotation Format

YOLO format (`.txt` files):
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
2 0.512 0.384 0.156 0.234 0.89
11 0.345 0.567 0.234 0.456 0.92
```

Where coordinates are normalized (0-1) relative to image dimensions.

## Architecture

### Project Structure
```
synthetic-data-generation-annotation/
├── app.py                          # Main Streamlit application
├── verify_pipeline.py              # Verification and testing script
├── requirements.txt                # Python dependencies
├── label_class.txt                 # Supported object classes
├── modules/
│   ├── __init__.py
│   ├── prompt_enhancement_module.py    # LLM-based prompt enhancement
│   ├── image_generation_module.py      # Diffusion model integration
│   └── annotation_module.py            # Grounding DINO annotation
└── pipeline_outputs/
    ├── generated_images/           # Generated image outputs
    └── annotations/                # YOLO format annotations
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Streamlit 1.28+ | Interactive web interface |
| LLM | LLaMA2 (via Ollama) | Prompt enhancement |
| Image Gen | Flux.1-dev / SD 1.4 | Diffusion-based image generation |
| Vision | Grounding DINO | Open-vocabulary object detection |
| Deep Learning | PyTorch 2.0+ | Model inference framework |
| Tokenization | tiktoken | Token counting and management |

## Configuration

### Model-Specific Settings

**Flux.1-dev Configuration** ([image_generation_module.py](image_generation_module.py)):
```python
FLUX_INFERENCE_STEPS = 30          # Denoising steps
FLUX_GUIDANCE_SCALE = 3.5          # Classifier-free guidance
FLUX_MAX_SEQUENCE_LENGTH = 512     # Token limit
```

**Stable Diffusion Configuration**:
```python
SD_INFERENCE_STEPS = 50
SD_GUIDANCE_SCALE = 7.5
SD_MAX_SEQUENCE_LENGTH = 72
```

**Annotation Configuration** ([annotation_module.py](annotation_module.py)):
```python
BOX_THRESHOLD = 0.20               # Detection threshold
TEXT_THRESHOLD = 0.25              # Text prompt matching threshold
NMS_IOU_THRESHOLD = 0.5            # IoU threshold for NMS
MIN_CONFIDENCE = 0.25              # Minimum confidence for export
```

### Environment Variables

Create a `.env` file or export these variables:

```bash
# Model paths
FLUX_PATH="/mnt/myssd/models/flux/..."
STABLE_DIFFUSION_PATH="/mnt/myssd/models/stable-diffusion-xl"
GROUNDING_DINO_PATH="/home/robot/Documents/VM_Annotation_Pipeline/Grounded-SAM-2"

# Hugging Face (optional, for model downloads)

# Device
CUDA_VISIBLE_DEVICES=0
```

## Authors and Acknowledgment

**Author:** Monika Chavan  

**Acknowledgments:**
- Black Forest Labs for [Flux.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- Stability AI for [Stable Diffusion](https://stability.ai/)
- Meta AI for [LLaMA2](https://ai.meta.com/llama/)
- IDEA Research for [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- Ollama team for local LLM inference framework

## License

This project is developed for academic research purposes.

**Note:** This project integrates several third-party models and frameworks, each with their own licenses:
- Flux.1-dev: [flux-1-dev-non-commercial-license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
- Stable Diffusion: CreativeML Open RAIL-M
- LLaMA2: [Meta LLaMA 2 Community License](https://ai.meta.com/llama/license/)
- Grounding DINO: Apache 2.0

Users must comply with all applicable licenses when using this pipeline.

## Project Status

**Status:** ✅ Active Development & Finalized

This pipeline has been successfully implemented and tested for synthetic data generation in industrial robotics scenarios.

**Last Updated:** December 2025  
**Version:** 1.0.0

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Repository:** [https://github.com/monikaits44/synthetic-data-generation-annotation](https://github.com/monikaits44/synthetic-data-generation-annotation)
