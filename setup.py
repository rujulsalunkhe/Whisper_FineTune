"""
Setup script for Whisper Fine-tuning Project.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "openai-whisper>=20231117",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "edge-tts>=6.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "jiwer>=3.0.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
        "click>=8.1.0"
    ]

setup(
    name="whisper-tech-finetuning",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Fine-tune Whisper models for better technical term recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper-tech-finetuning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "monitoring": [
            "tensorboard>=2.14.0",
            "wandb>=0.16.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "whisper-generate-data=scripts.generate_training_data:main",
            "whisper-train=scripts.train_model:main",
            "whisper-evaluate=scripts.evaluate_model:main",
            "whisper-infer=scripts.inference:main",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
)