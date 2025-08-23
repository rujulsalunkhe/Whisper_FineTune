#!/usr/bin/env python3
"""
Training script for fine-tuning Whisper models on technical terms.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.whisper_trainer import WhisperTechTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Whisper model for technical terms')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model_size', 
        type=str, 
        choices=['large-v3', 'large-v3-turbo'],
        default='large-v3',
        help='Whisper model size to fine-tune'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        help='Training batch size (overrides config)'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        help='Output directory for trained model (overrides config)'
    )
    parser.add_argument(
        '--resume_from_checkpoint', 
        type=str, 
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--no_eval', 
        action='store_true',
        help='Skip evaluation during training'
    )
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Run setup without actual training'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Update configuration with command line arguments."""
    # Update model name based on model_size
    model_mapping = {
        'large-v3': 'openai/whisper-large-v3',
        'large-v3-turbo': 'openai/whisper-large-v3-turbo'
    }
    config['model']['name'] = model_mapping[args.model_size]
    
    # Override training parameters if provided
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    else:
        # Update output directory with model size
        base_dir = config['training']['output_dir']
        config['training']['output_dir'] = f"{base_dir}-{args.model_size}"
    
    return config


def validate_data_paths(config: dict) -> bool:
    """Validate that required data files exist."""
    required_files = [
        config['data']['train_transcripts'],
        config['data']['val_transcripts'] if not config.get('no_eval', False) else None
    ]
    
    required_dirs = [
        config['data']['train_audio_dir'],
        config['data']['val_audio_dir'] if not config.get('no_eval', False) else None
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path and not Path(file_path).exists():
            missing_files.append(file_path)
            
    for dir_path in required_dirs:
        if dir_path and not Path(dir_path).exists():
            missing_files.append(dir_path)
    
    if missing_files:
        logger.error("Missing required data files/directories:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        logger.error("\nPlease run 'python scripts/generate_training_data.py' first to create training data.")
        return False
        
    return True


def setup_output_directory(output_dir: str) -> None:
    """Create output directory and subdirectories."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "checkpoints").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    config['no_eval'] = args.no_eval
    
    # Validate data paths
    if not validate_data_paths(config):
        sys.exit(1)
    
    # Setup output directory
    setup_output_directory(config['training']['output_dir'])
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"  Model: {config['model']['name']}")
    logger.info(f"  Epochs: {config['training']['num_epochs']}")
    logger.info(f"  Batch Size: {config['training']['batch_size']}")
    logger.info(f"  Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"  Output Directory: {config['training']['output_dir']}")
    logger.info(f"  Use LoRA: {config['training']['use_lora']}")
    
    if args.dry_run:
        logger.info("Dry run completed successfully. Configuration is valid.")
        return
    
    try:
        # Initialize trainer
        logger.info("Initializing Whisper trainer...")
        trainer = WhisperTechTrainer(config)
        trainer.setup_model()
        
        # Save configuration
        trainer.save_config()
        
        # Load datasets
        logger.info("Loading training dataset...")
        train_dataset = trainer.load_dataset(
            config['data']['train_audio_dir'],
            config['data']['train_transcripts']
        )
        train_dataset = trainer.prepare_dataset(train_dataset)
        
        eval_dataset = None
        if not args.no_eval:
            logger.info("Loading validation dataset...")
            eval_dataset = trainer.load_dataset(
                config['data']['val_audio_dir'],
                config['data']['val_transcripts']
            )
            eval_dataset = trainer.prepare_dataset(eval_dataset)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation samples: {len(eval_dataset)}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(train_dataset, eval_dataset)
        
        # Final evaluation
        if eval_dataset:
            logger.info("Running final evaluation...")
            results = trainer.evaluate(eval_dataset)
            
            # Save results
            results_path = Path(config['training']['output_dir']) / "final_results.json"
            import json
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {config['training']['output_dir']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()