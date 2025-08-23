#!/usr/bin/env python3
"""
Inference script for testing fine-tuned Whisper models.
"""

import os
import sys
import argparse
import yaml
import logging
import time
from pathlib import Path
import torch
import librosa
import soundfile as sf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model.whisper_trainer import WhisperTechTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for better output
console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test fine-tuned Whisper model')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/whisper-tech-finetuned-large-v3',
        help='Path to fine-tuned model directory'
    )
    parser.add_argument(
        '--audio_path',
        type=str,
        help='Path to single audio file for transcription'
    )
    parser.add_argument(
        '--audio_dir',
        type=str,
        help='Path to directory containing audio files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='Save transcriptions to file'
    )
    parser.add_argument(
        '--compare_original',
        action='store_true',
        help='Compare with original Whisper model'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark on technical terms'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode for testing'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def transcribe_audio(model_trainer, audio_path: str) -> tuple:
    """Transcribe a single audio file and measure performance."""
    start_time = time.time()
    transcription = model_trainer.inference(audio_path)
    inference_time = time.time() - start_time
    
    # Get audio duration
    audio, sr = librosa.load(audio_path, sr=None)
    duration = len(audio) / sr
    
    return transcription, inference_time, duration


def compare_models(fine_tuned_trainer, original_trainer, audio_path: str):
    """Compare fine-tuned model with original Whisper."""
    console.print("\n[bold blue]Comparing Models[/bold blue]")
    
    # Transcribe with fine-tuned model
    ft_transcription, ft_time, duration = transcribe_audio(fine_tuned_trainer, audio_path)
    
    # Transcribe with original model
    orig_transcription, orig_time, _ = transcribe_audio(original_trainer, audio_path)
    
    # Create comparison table
    table = Table(title=f"Model Comparison - Audio Duration: {duration:.2f}s")
    table.add_column("Model", style="cyan")
    table.add_column("Transcription", style="magenta")
    table.add_column("Time (s)", style="green")
    table.add_column("RTF", style="yellow")
    
    table.add_row(
        "Fine-tuned",
        ft_transcription,
        f"{ft_time:.2f}",
        f"{ft_time/duration:.2f}x"
    )
    table.add_row(
        "Original",
        orig_transcription,
        f"{orig_time:.2f}",
        f"{orig_time/duration:.2f}x"
    )
    
    console.print(table)


def run_benchmark(model_trainer, terms_data: dict):
    """Run benchmark on technical terms."""
    console.print("\n[bold blue]Running Technical Terms Benchmark[/bold blue]")
    
    # Create test sentences with technical terms
    test_sentences = []
    all_terms = []
    
    for category in terms_data['technical_terms'].values():
        if isinstance(category, list):
            for term_data in category:
                term = term_data['term']
                all_terms.append(term)
                test_sentences.append(f"I'm using {term} for this project")
                test_sentences.append(f"Can you help me configure {term}?")
    
    # Note: In a real benchmark, you'd have pre-recorded audio files
    # This is a placeholder for the benchmark structure
    console.print(f"[yellow]Benchmark would test {len(test_sentences)} sentences with {len(all_terms)} technical terms[/yellow]")
    console.print("[yellow]Note: Actual benchmark requires pre-recorded test audio files[/yellow]")


def interactive_mode(model_trainer):
    """Interactive mode for testing transcriptions."""
    console.print("\n[bold green]Interactive Mode[/bold green]")
    console.print("Enter audio file paths to transcribe (type 'quit' to exit)")
    
    while True:
        try:
            audio_path = console.input("\n[cyan]Audio file path: [/cyan]").strip()
            
            if audio_path.lower() in ['quit', 'exit', 'q']:
                break
                
            if not audio_path:
                continue
                
            audio_file = Path(audio_path)
            if not audio_file.exists():
                console.print(f"[red]File not found: {audio_path}[/red]")
                continue
                
            console.print(f"\n[yellow]Transcribing: {audio_file.name}[/yellow]")
            transcription, inference_time, duration = transcribe_audio(model_trainer, audio_path)
            
            # Display results in a panel
            result_text = f"""[bold]Transcription:[/bold] {transcription}
[bold]Duration:[/bold] {duration:.2f}s
[bold]Inference Time:[/bold] {inference_time:.2f}s
[bold]Real-time Factor:[/bold] {inference_time/duration:.2f}x"""
            
            console.print(Panel(result_text, title="Results", border_style="green"))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


def process_audio_directory(model_trainer, audio_dir: str, output_file: str = None):
    """Process all audio files in a directory."""
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        console.print(f"[red]Directory not found: {audio_dir}[/red]")
        return
    
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(audio_path.glob(f'*{ext}'))
    
    if not audio_files:
        console.print(f"[red]No audio files found in: {audio_dir}[/red]")
        return
    
    console.print(f"\n[bold blue]Processing {len(audio_files)} audio files[/bold blue]")
    
    results = []
    total_duration = 0
    total_inference_time = 0
    
    for i, audio_file in enumerate(audio_files, 1):
        console.print(f"[cyan]{i}/{len(audio_files)}: {audio_file.name}[/cyan]")
        
        try:
            transcription, inference_time, duration = transcribe_audio(model_trainer, str(audio_file))
            
            results.append({
                'file': audio_file.name,
                'transcription': transcription,
                'duration': duration,
                'inference_time': inference_time
            })
            
            total_duration += duration
            total_inference_time += inference_time
            
            console.print(f"  [green]✓ {transcription}[/green]")
            
        except Exception as e:
            console.print(f"  [red]✗ Error: {str(e)}[/red]")
    
    # Display summary
    avg_rtf = total_inference_time / total_duration if total_duration > 0 else 0
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Total files processed: {len(results)}")
    console.print(f"  Total audio duration: {total_duration:.2f}s")
    console.print(f"  Total inference time: {total_inference_time:.2f}s")
    console.print(f"  Average RTF: {avg_rtf:.2f}x")
    
    # Save results if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to: {output_file}[/green]")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        console.print(f"[red]Model not found: {args.model_path}[/red]")
        console.print("[yellow]Please train a model first with: python scripts/train_model.py[/yellow]")
        sys.exit(1)
    
    console.print(Panel(f"Loading fine-tuned model from: {args.model_path}", style="blue"))
    
    try:
        # Load fine-tuned model
        fine_tuned_trainer = WhisperTechTrainer.load_trained_model(args.model_path, config)
        console.print("[green]✓ Fine-tuned model loaded successfully[/green]")
        
        # Load original model for comparison if requested
        original_trainer = None
        if args.compare_original:
            console.print("Loading original Whisper model for comparison...")
            original_config = config.copy()
            original_trainer = WhisperTechTrainer(original_config)
            original_trainer.setup_model()
            console.print("[green]✓ Original model loaded successfully[/green]")
        
        # Load technical terms for benchmark
        terms_data = None
        if args.benchmark:
            terms_file = Path("data/tech_terms.json")
            if terms_file.exists():
                import json
                with open(terms_file, 'r') as f:
                    terms_data = json.load(f)
        
        # Process based on arguments
        if args.interactive:
            interactive_mode(fine_tuned_trainer)
            
        elif args.audio_path:
            audio_file = Path(args.audio_path)
            if not audio_file.exists():
                console.print(f"[red]Audio file not found: {args.audio_path}[/red]")
                sys.exit(1)
            
            console.print(f"\n[bold blue]Transcribing: {audio_file.name}[/bold blue]")
            
            if args.compare_original and original_trainer:
                compare_models(fine_tuned_trainer, original_trainer, args.audio_path)
            else:
                transcription, inference_time, duration = transcribe_audio(fine_tuned_trainer, args.audio_path)
                
                result_text = f"""[bold]File:[/bold] {audio_file.name}
[bold]Transcription:[/bold] {transcription}
[bold]Duration:[/bold] {duration:.2f}s
[bold]Inference Time:[/bold] {inference_time:.2f}s
[bold]Real-time Factor:[/bold] {inference_time/duration:.2f}x"""
                
                console.print(Panel(result_text, title="Transcription Result", border_style="green"))
                
        elif args.audio_dir:
            process_audio_directory(fine_tuned_trainer, args.audio_dir, args.output_file)
            
        elif args.benchmark and terms_data:
            run_benchmark(fine_tuned_trainer, terms_data)
            
        else:
            console.print("[yellow]No input specified. Use --help for options or --interactive for interactive mode[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()