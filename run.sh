#!/bin/bash
# Complete setup and training script for Whisper Technical Terms Fine-tuning

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU
check_gpu() {
    if command_exists nvidia-smi; then
        print_status "Checking GPU availability..."
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        return 0
    else
        print_warning "nvidia-smi not found. GPU may not be available."
        return 1
    fi
}

# Function to check Python environment
check_python() {
    print_status "Checking Python environment..."
    
    # Check Python version
    python_version=$(python --version 2>&1 | awk '{print $2}')
    print_status "Python version: $python_version"
    
    # Check if in virtual environment
    if [[ -n "$VIRTUAL_ENV" ]] || [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        print_success "Virtual environment detected: ${VIRTUAL_ENV:-$CONDA_DEFAULT_ENV}"
    else
        print_warning "No virtual environment detected. Consider using one."
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch (you may need to adjust for your CUDA version)
    print_status "Installing PyTorch..."
    if check_gpu >/dev/null 2>&1; then
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        print_warning "Installing CPU-only PyTorch"
        pip install torch torchaudio
    fi
    
    # Install other dependencies
    print_status "Installing project dependencies..."
    pip install -r requirements.txt
    
    print_success "Dependencies installed successfully"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    python << EOF
import sys
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
    
    import whisper
    print(f"✅ Whisper available")
    
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
    
    import datasets
    print(f"✅ Datasets {datasets.__version__}")
    
    import peft
    print(f"✅ PEFT available")
    
    print("🎉 All core dependencies verified!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Function to setup project structure
setup_project() {
    print_status "Setting up project structure..."
    
    # Create necessary directories
    mkdir -p data/{train_audio,val_audio}
    mkdir -p models
    mkdir -p evaluation_results
    mkdir -p logs
    
    # Create __init__.py files
    touch data/__init__.py
    touch src/__init__.py
    touch src/model/__init__.py
    touch src/utils/__init__.py
    touch src/evaluation/__init__.py
    touch config/__init__.py
    
    print_success "Project structure created"
}

# Function to generate training data
generate_data() {
    print_status "Generating training data..."
    
    if [ ! -f "data/tech_terms.json" ]; then
        print_error "tech_terms.json not found in data directory"
        exit 1
    fi
    
    # Check if data already exists
    if [ -f "data/train_transcripts.json" ] && [ -f "data/val_transcripts.json" ]; then
        read -p "Training data already exists. Regenerate? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping data generation"
            return 0
        fi
    fi
    
    # Generate data
    python scripts/generate_training_data.py --samples_per_term 15
    
    if [ $? -eq 0 ]; then
        print_success "Training data generated successfully"
    else
        print_error "Data generation failed"
        exit 1
    fi
}

# Function to train model
train_model() {
    local model_size=${1:-"large-v3"}
    local epochs=${2:-10}
    
    print_status "Training Whisper $model_size model for $epochs epochs..."
    
    # Check if trained model already exists
    model_dir="models/whisper-tech-finetuned-$model_size"
    if [ -d "$model_dir" ]; then
        read -p "Model already exists at $model_dir. Retrain? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping training"
            return 0
        fi
    fi
    
    # Start training
    python scripts/train_model.py \
        --model_size "$model_size" \
        --epochs "$epochs" \
        --batch_size 8 \
        --learning_rate 1e-5
    
    if [ $? -eq 0 ]; then
        print_success "Training completed successfully"
        print_status "Model saved to: $model_dir"
    else
        print_error "Training failed"
        exit 1
    fi
}

# Function to evaluate model
evaluate_model() {
    local model_size=${1:-"large-v3"}
    
    print_status "Evaluating trained model..."
    
    model_dir="models/whisper-tech-finetuned-$model_size"
    if [ ! -d "$model_dir" ]; then
        print_error "Trained model not found at $model_dir"
        print_error "Please train the model first"
        exit 1
    fi
    
    python scripts/evaluate_model.py \
        --model_path "$model_dir" \
        --compare_original \
        --detailed_report
    
    if [ $? -eq 0 ]; then
        print_success "Evaluation completed"
        print_status "Results saved to: evaluation_results/"
    else
        print_error "Evaluation failed"
        exit 1
    fi
}

# Function to run inference demo
run_demo() {
    local model_size=${1:-"large-v3"}
    
    print_status "Running inference demo..."
    
    model_dir="models/whisper-tech-finetuned-$model_size"
    if [ ! -d "$model_dir" ]; then
        print_error "Trained model not found at $model_dir"
        exit 1
    fi
    
    print_status "Starting interactive inference mode"
    print_status "You can enter audio file paths to test transcription"
    print_status "Type 'quit' to exit"
    
    python scripts/inference.py \
        --model_path "$model_dir" \
        --interactive
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  setup                 - Setup project and install dependencies"
    echo "  generate-data         - Generate training data using TTS"
    echo "  train [MODEL] [EPOCHS] - Train model (default: large-v3, 10 epochs)"
    echo "  evaluate [MODEL]      - Evaluate trained model"
    echo "  demo [MODEL]          - Run interactive inference demo"
    echo "  full-pipeline [MODEL] - Run complete pipeline (setup -> train -> evaluate)"
    echo "  clean                 - Clean generated data and models"
    echo
    echo "Models: large-v3, large-v3-turbo"
    echo
    echo "Examples:"
    echo "  $0 setup                    # Setup project"
    echo "  $0 full-pipeline large-v3   # Run complete pipeline with large-v3"
    echo "  $0 train large-v3-turbo 5   # Train turbo model for 5 epochs"
    echo "  $0 demo large-v3            # Run demo with large-v3 model"
}

# Function to run full pipeline
run_full_pipeline() {
    local model_size=${1:-"large-v3"}
    local epochs=${2:-10}
    
    print_status "Running full pipeline for $model_size model"
    
    setup_project
    generate_data
    train_model "$model_size" "$epochs"
    evaluate_model "$model_size"
    
    print_success "Full pipeline completed!"
    print_status "To test the model, run: $0 demo $model_size"
}

# Function to clean up
clean_up() {
    print_warning "This will remove all generated data and models"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/train_audio data/val_audio
        rm -f data/train_transcripts.json data/val_transcripts.json
        rm -rf models/whisper-tech-finetuned-*
        rm -rf evaluation_results
        rm -rf logs
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main script logic
main() {
    local command=${1:-help}
    
    case $command in
        "setup")
            print_status "Setting up Whisper Technical Terms Fine-tuning project"
            check_python
            check_gpu
            install_dependencies
            verify_installation
            setup_project
            print_success "Setup completed successfully!"
            print_status "Next steps:"
            print_status "1. Run: $0 generate-data"
            print_status "2. Run: $0 train"
            print_status "3. Run: $0 evaluate"
            ;;
        "generate-data")
            generate_data
            ;;
        "train")
            train_model "${2:-large-v3}" "${3:-10}"
            ;;
        "evaluate")
            evaluate_model "${2:-large-v3}"
            ;;
        "demo")
            run_demo "${2:-large-v3}"
            ;;
        "full-pipeline")
            run_full_pipeline "${2:-large-v3}" "${3:-10}"
            ;;
        "clean")
            clean_up
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"