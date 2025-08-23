#!/usr/bin/env python3
"""
Generate training data for Whisper fine-tuning using TTS and technical terms.
"""

import os
import sys
import argparse
import asyncio
import yaml
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data.audio_generator import TechTermAudioGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate training data for Whisper fine-tuning')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data',
        help='Output directory for generated data'
    )
    parser.add_argument(
        '--terms_file', 
        type=str, 
        default='data/tech_terms.json',
        help='Path to technical terms JSON file'
    )
    parser.add_argument(
        '--samples_per_term', 
        type=int, 
        default=15,
        help='Number of audio samples to generate per term'
    )
    parser.add_argument(
        '--voices', 
        type=str, 
        nargs='+',
        default=[
            'en-US-AriaNeural',
            'en-US-JennyNeural', 
            'en-US-GuyNeural',
            'en-GB-SoniaNeural'
        ],
        help='List of TTS voices to use'
    )
    parser.add_argument(
        '--train_split', 
        type=float, 
        default=0.8,
        help='Proportion of data for training (rest for validation)'
    )
    parser.add_argument(
        '--skip_existing', 
        action='store_true',
        help='Skip generation if output files already exist'
    )
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Show what would be generated without actually generating'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import edge_tts
        logger.info("✓ edge-tts available")
    except ImportError:
        logger.error("✗ edge-tts not available. Install with: pip install edge-tts")
        return False
        
    try:
        import pydub
        logger.info("✓ pydub available")
    except ImportError:
        logger.error("✗ pydub not available. Install with: pip install pydub")
        return False
        
    try:
        import soundfile
        logger.info("✓ soundfile available")
    except ImportError:
        logger.error("✗ soundfile not available. Install with: pip install soundfile")
        return False
        
    return True


def estimate_generation_time(config: dict, args: argparse.Namespace) -> str:
    """Estimate how long generation will take."""
    # Load technical terms to count them
    terms_file = Path(args.terms_file)
    if not terms_file.exists():
        return "Unknown (terms file not found)"
        
    import json
    with open(terms_file, 'r') as f:
        terms_data = json.load(f)
    
    # Count terms
    total_terms = 0
    for category in terms_data['technical_terms'].values():
        if isinstance(category, list):
            total_terms += len(category)
    
    # Estimate time
    samples_per_term = args.samples_per_term
    num_voices = len(args.voices)
    total_samples = total_terms * samples_per_term
    
    # Rough estimate: 2-3 seconds per sample (TTS + processing)
    estimated_seconds = total_samples * 2.5
    estimated_minutes = estimated_seconds / 60
    
    return f"~{estimated_minutes:.1f} minutes ({total_samples} total samples)"


async def main():
    """Main data generation function."""
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them first.")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    audio_config = config.get('audio_generation', {})
    audio_config.update({
        'voices': args.voices,
        'samples_per_term': args.samples_per_term,
        'sample_rate': config.get('data', {}).get('sampling_rate', 16000),
        'apply_noise': config.get('data', {}).get('apply_noise', True),
        'speed_perturbation': config.get('data', {}).get('speed_perturbation', True),
        'templates': config.get('audio_generation', {}).get('templates', [
            "I'm using {term} for this project",
            "Let's install {term} first",
            "The {term} documentation is helpful",
            "Can you help me with {term}?",
            "I need to configure {term} properly"
        ])
    })
    
    # Setup paths
    output_dir = Path(args.output_dir)
    terms_file = Path(args.terms_file)
    
    # Validate inputs
    if not terms_file.exists():
        logger.error(f"Technical terms file not found: {terms_file}")
        logger.error("Please ensure the tech_terms.json file exists in the data directory")
        sys.exit(1)
    
    # Check if output already exists
    train_transcripts = output_dir / "train_transcripts.json"
    val_transcripts = output_dir / "val_transcripts.json"
    
    if args.skip_existing and train_transcripts.exists() and val_transcripts.exists():
        logger.info("Training data already exists and --skip_existing specified. Exiting.")
        return
    
    # Estimate generation time
    estimated_time = estimate_generation_time(config, args)
    logger.info(f"Estimated generation time: {estimated_time}")
    
    if args.dry_run:
        logger.info("Dry run - showing what would be generated:")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Terms file: {terms_file}")
        logger.info(f"  Samples per term: {args.samples_per_term}")
        logger.info(f"  Voices: {args.voices}")
        logger.info(f"  Train/val split: {args.train_split:.1%}")
        logger.info(f"  Estimated time: {estimated_time}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize generator
        logger.info("Initializing audio generator...")
        generator = TechTermAudioGenerator(audio_config)
        generator.load_technical_terms(str(terms_file))
        
        # Generate audio samples
        logger.info("Starting audio generation...")
        logger.info(f"This may take a while ({estimated_time})")
        
        dataset = await generator.generate_audio_samples(
            str(output_dir), 
            args.train_split
        )
        
        # Save dataset metadata
        generator.save_dataset_metadata(dataset, str(output_dir))
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("GENERATION COMPLETE")
        logger.info("="*50)
        logger.info(f"✓ Generated {len(dataset['train'])} training samples")
        logger.info(f"✓ Generated {len(dataset['validation'])} validation samples")
        logger.info(f"✓ Audio files saved to: {output_dir}")
        logger.info(f"✓ Metadata saved to: {train_transcripts} and {val_transcripts}")
        
        # Verify generated files
        train_audio_dir = output_dir / "train_audio"
        val_audio_dir = output_dir / "val_audio"
        
        train_files = list(train_audio_dir.glob("*.wav")) if train_audio_dir.exists() else []
        val_files = list(val_audio_dir.glob("*.wav")) if val_audio_dir.exists() else []
        
        logger.info(f"✓ {len(train_files)} training audio files")
        logger.info(f"✓ {len(val_files)} validation audio files")
        
        # Calculate total size
        total_size = 0
        for audio_file in train_files + val_files:
            total_size += audio_file.stat().st_size
        total_size_mb = total_size / (1024 * 1024)
        
        logger.info(f"✓ Total dataset size: {total_size_mb:.1f} MB")
        logger.info("\nYou can now run training with:")
        logger.info("python scripts/train_model.py")
        
    except KeyboardInterrupt:
        logger.info("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())