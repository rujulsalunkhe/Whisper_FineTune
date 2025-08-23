# Whisper Technical Terms Fine-tuning

A comprehensive project for fine-tuning OpenAI's Whisper models (Large v3 and Large v3 Turbo) to improve recognition of technical terminology including developer tools, programming languages, and cloud platforms.

## Features

- **Fine-tuning Support**: Works with Whisper Large v3 and Large v3 Turbo
- **Technical Term Focus**: Optimized for recognizing technical jargon like "maven", "github", "kubernetes", etc.
- **Synthetic Data Generation**: Automatically generates training data with text-to-speech
- **Audio Augmentation**: Applies noise, speed, pitch, and volume variations
- **Comprehensive Evaluation**: Multiple metrics including WER, technical term accuracy, and more
- **Easy Training**: Simple command-line interface for training and evaluation
- **Model Comparison**: Compare fine-tuned models with original Whisper

## Technical Terms Supported

The model is optimized for recognizing:
- Build tools: `maven`, `mvn`, `npm`, `gradle`
- Version control: `git`, `github`, `gitlab`
- Cloud platforms: `aws`, `azure`, `gcp`, `kubernetes`
- AI/ML: `openai`, `chatgpt`, `llm`, `groq`, `grok`
- Programming: `react`, `angular`, `vue`, `docker`
- And many more...

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- FFmpeg for audio processing

### Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd whisper-tech-finetuning
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the project:**
```bash
pip install -e .
```

## Quick Start

### 1. Generate Training Data

```bash
python scripts/generate_synthetic_data.py --num-samples 1000
```

### 2. Train the Model

**For Whisper Large v3:**
```bash
python scripts/train.py \
    --model-name "openai/whisper-large-v3" \
    --generate-dataset \
    --num-samples 1000 \
    --output-dir ./models/whisper-large-v3-tech
```

**For Whisper Large v3 Turbo:**
```bash
python scripts/train.py \
    --model-name "openai/whisper-large-v3-turbo" \
    --generate-dataset \
    --num-samples 1000 \
    --output-dir ./models/whisper-large-v3-turbo-tech
```

### 3. Test the Model

```bash
python scripts/test_model.py \
    --model-path ./models/whisper-large-v3-tech/best_model \
    --audio-file path/to/your/audio.wav \
    --compare-original
```

## Detailed Usage

### Configuration

The training behavior is controlled by `config/config.yaml`. Key parameters:

```yaml
model:
  name: "openai/whisper-large-v3"  # Model to fine-tune
  language: "en"
  task: "transcribe"

training:
  batch_size: 16
  learning_rate: 1e-5
  max_steps: 5000
  eval_steps: 1000

data:
  max_input_length: 30  # seconds
  sampling_rate: 16000
  augmentation:
    enabled: true
    noise_factor: 0.005
```

### Training Options

**Basic Training:**
```bash
python scripts/train.py --config config/config.yaml
```

**Advanced Training:**
```bash
python scripts/train.py \
    --model-name "openai/whisper-large-v3" \
    --generate-dataset \
    --num-samples 2000 \
    --output-dir ./models/custom-model \
    --no-wandb \
    --no-augment
```

**Resume Training:**
```bash
python scripts/train.py \
    --resume-from-checkpoint ./models/checkpoint-1000 \
    --dataset-path data/synthetic/dataset.json
```

### Evaluation Options

**Single Audio File:**
```bash
python scripts/test_model.py \
    --model-path ./models/best_model \
    --audio-file recording.wav
```

**Batch Processing:**
```bash
python scripts/test_model.py \
    --model-path ./models/best_model \
    --audio-dir ./test_audio/ \
    --output-file results.json
```

**Dataset Evaluation:**
```bash
python scripts/test_model.py \
    --model-path ./models/best_model \
    --test-dataset data/synthetic/test_dataset.json \
    --compare-original \
    --output-file evaluation_results.json
```

## Project Structure

```
whisper-tech-finetuning/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup.py                   # Package setup
├── config/
│   ├── config.yaml            # Training configuration
│   └── training_config.py     # Config classes
├── data/
│   ├── prepare_dataset.py     # Dataset generation
│   ├── tech_terms.txt         # Technical terms list
│   └── synthetic/             # Generated datasets
├── src/
│   ├── models/
│   │   └── whisper_model.py   # Custom Whisper model
│   ├── training/
│   │   ├── trainer.py         # Training logic
│   │   └── dataset.py         # Dataset classes
│   ├── evaluation/
│   │   └── metrics.py         # Evaluation metrics
│   └── utils/
│       └── logger.py          # Logging utilities
├── scripts/
│   ├── train.py               # Training script
│   ├── test_model.py          # Testing script
│   └── generate_synthetic_data.py
└── models/                    # Saved models
```

## Training Process

1. **Data Preparation**: Generate synthetic audio samples using TTS
2. **Vocabulary Extension**: Add technical terms to Whisper's tokenizer
3. **Fine-tuning**: Train with technical term-focused data
4. **Evaluation**: Measure improvement in technical term recognition
5. **Model Selection**: Save the best performing checkpoint

## Evaluation Metrics

- **WER (Word Error Rate)**: Overall transcription accuracy
- **CER (Character Error Rate)**: Character-level accuracy
- **Technical Term Accuracy**: Accuracy specifically for technical terms
- **BLEU Score**: Translation-like quality metric
- **Exact Match**: Percentage of perfect transcriptions

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070 or equivalent)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space

### Recommended Requirements
- **GPU**: 16GB+ VRAM (RTX 4080/4090, V100, A100)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space on SSD

### Training Time Estimates
- **Whisper Large v3**: ~2-4 hours on RTX 4090
- **Whisper Large v3 Turbo**: ~1-2 hours on RTX 4090

## Customization

### Adding New Technical Terms

1. **Edit the terms list:**
```bash
nano data/tech_terms.txt
```

2. **Update configuration:**
```yaml
technical_terms:
  - your_new_term
  - another_term
```

3. **Regenerate dataset:**
```bash
python scripts/generate_synthetic_data.py --num-samples 1000
```

### Custom Audio Data

If you have your own audio data:

1. **Prepare your dataset** in this format:
```json
[
  {
    "audio_path": "path/to/audio1.wav",
    "text": "The transcription text",
    "duration": 5.2
  }
]
```

2. **Train with custom data:**
```bash
python scripts/train.py --dataset-path your_dataset.json
```

## Monitoring Training

### Weights & Biases Integration

The project integrates with W&B for experiment tracking:

```bash
# Login to W&B (first time only)
wandb login

# Training with W&B logging (enabled by default)
python scripts/train.py --model-name "openai/whisper-large-v3"
```

### Local Monitoring

Monitor training progress through:
- Console logs with detailed metrics
- Local tensorboard logs (if enabled)
- JSON training history saved with each model

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config
   - Enable `gradient_checkpointing`
   - Use gradient accumulation

2. **Audio Processing Errors**
   - Install FFmpeg: `sudo apt install ffmpeg`
   - Check audio file formats (WAV recommended)

3. **TTS Generation Fails**
   - Install espeak: `sudo apt install espeak`
   - Check internet connection for gTTS

4. **Model Loading Issues**
   - Ensure sufficient disk space
   - Check Hugging Face authentication if needed

### Performance Optimization

```python
# In config.yaml
training:
  batch_size: 8          # Reduce if OOM
  gradient_accumulation_steps: 2  # Effective batch size = 16
  
hardware:
  mixed_precision: true   # Enable FP16
  gradient_checkpointing: true  # Save memory
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{whisper-tech-finetuning,
  title={Fine-tuning Whisper for Technical Term Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/whisper-tech-finetuning}
}
```

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the transformers library
- The open-source community for various tools and libraries