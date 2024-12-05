# CPU-Optimized AI Assistant

An efficient, open-source AI assistant designed to run smoothly on CPU-only systems. This project uses lightweight models and optimized inference techniques to provide AI capabilities without requiring expensive GPU hardware.

## Features

- Local inference using optimized models
- Text generation and conversation
- Efficient CPU-based processing
- Easy-to-use API interface
- Modular design for easy model switching

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models:
The assistant will automatically download required models on first run.

## Usage

Start the assistant:
```bash
python src/main.py
```

## Models Used

- Text Generation: GPT4All-J or Llama CPP models
- Embeddings: All-MiniLM-L6-v2
- Text Classification: DistilBERT base uncased

## Architecture

The assistant uses a modular architecture with the following components:
- Model Manager: Handles model loading and optimization
- Inference Engine: Manages text generation and processing
- API Server: Provides HTTP interface for interactions
- Memory System: Manages conversation context

## Performance Optimization

- Quantized models for reduced memory usage
- Efficient batching and caching
- Optimized inference settings for CPU
- Memory-efficient conversation handling
