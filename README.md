# (Simulation) Smart Home Assistant using LLMs

This repository contains a **simulated smart home environment** designed for educational purposes. It serves as a practical example of how Large Language Models (LLMs) can be applied to complex tasks like home automation control.

### Purpose
- **Learning Tool**: Demonstrate LLM capabilities in understanding and processing natural language commands
- **Simulation Only**: This is not a production-ready system for real home automation
- **Experimental**: Showcases different approaches (LLM vs regex) for command interpretation
- **Multilingual**: Supports both English and Spanish to demonstrate language flexibility

### Features Used
- Natural Language Processing with various LLMs (Phi-4, Qwen, Llama)
- Voice Processing (Whisper ASR)
- Text-to-Speech Synthesis (MMS-TTS)
- Pattern Matching and State Management
- Real-time UI with Gradio

## Project Structure

```
iein/
├── assistant.py      # Main application interface and Gradio UI implementation
│                     # Manages user interactions, model orchestration, and UI components
│
├── home.py           # Smart home simulation and state management
│                     # Handles device states, command processing, and event logging
│                     # Supports both regex and LLM-based command interpretation
│
├── models.py         # AI model implementations and interfaces
│                     # Contains:
│                     # - BaseLLM: Abstract base class for all AI models
│                     # - WhisperASR: Speech recognition (OpenAI Whisper)
│                     # - ChatLLM: Language model interface (supports multiple LLMs)
│                     # - VITTS: Text-to-speech synthesis (Meta MMS-TTS)
│
└── requirements.txt  # Project dependencies
```

## System Requirements

### Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB free space

### Recommended Requirements
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM
- Storage: 20GB+ free space

## Quick Start

### 1. Google Colab Setup

For a quick test without local installation, use our Colab notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Rjtl0juRd6j0u9JKREsnqh_CeXRAGbzF?usp=sharing)

### 2. Local Setup

#### Option A: Using UV (Recommended)
[uv](https://docs.astral.sh/uv/pip/) is a modern Python package installer. We recommend installing it with [pipx](https://pipx.pypa.io/stable/installation/).

```bash
# Create and activate virtual environment using uv 
uv venv my-uvenv --python 3.10
source my-uvenv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install torch dependencies (CUDA version >=12.5)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### Option B: Using venv
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the assistant
python assistant.py
```

