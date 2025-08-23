#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Whisper models.
"""

import os
import sys
import argparse
import yaml
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import jiwer

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
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Whisper model')
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/whisper-tech-finetuned-large-v3',
        help='Path to fine-tuned model directory'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        help='Path to test dataset JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--compare_original',
        action='store_true',
        help='Compare with original Whisper model'
    )
    parser.add_argument(
        '--detailed_report',
        action='store_true',
        help='Generate detailed evaluation report'
    )
    parser.add_argument(
        '--tech_terms_only',
        action='store_true',
        help='Evaluate only on samples containing technical terms'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_technical_terms(terms_file: str) -> List[str]:
    """Load list of technical terms."""
    with open(terms_file, 'r') as f:
        terms_data = json.load(f)
    
    all_terms = []
    for category in terms_data['technical_terms'].values():
        if isinstance(category, list):
            for term_data in category:
                all_terms.append(term_data['term'].lower())
                all_terms.extend([v.lower() for v in term_data.get('variations', [])])
    
    return list(set(all_terms))


def contains_tech_terms(text: str, tech_terms: List[str]) -> bool:
    """Check if text contains any technical terms."""
    text_lower = text.lower()
    return any(term in text_lower for term in tech_terms)


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    # Word Error Rate
    wer = jiwer.wer(references, predictions)
    
    # Character Error Rate
    cer = jiwer.cer(references, references)
    
    # BLEU score
    try:
        from nltk.translate.bleu_score import corpus_bleu
        # Tokenize for BLEU calculation
        ref_tokens = [[ref.split()] for ref in references]
        pred_tokens = [pred.split() for pred in predictions]
        bleu = corpus_bleu(ref_tokens, pred_tokens)
    except ImportError:
        bleu = 0.0
    
    return {
        'wer': wer,
        'cer': cer,
        'bleu': bleu,
        'accuracy': 1 - wer  # Simple accuracy metric
    }


def calculate_tech_term_metrics(predictions: List[str], references: List[str], tech_terms: List[str]) -> Dict[str, Any]:
    """Calculate metrics specifically for technical terms."""
    term_stats = {}
    total_terms = 0
    correct_terms = 0
    
    for pred, ref in zip(predictions, references):
        pred_lower = pred.lower()
        ref_lower = ref.lower()
        
        for term in tech_terms:
            if term in ref_lower:
                total_terms += 1
                if term not in term_stats:
                    term_stats[term] = {'total': 0, 'correct': 0}
                term_stats[term]['total'] += 1
                
                if term in pred_lower:
                    correct_terms += 1
                    term_stats[term]['correct'] += 1
    
    # Calculate per-term accuracy
    term_accuracies = {}
    for term, stats in term_stats.items():
        term_accuracies[term] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
    
    overall_accuracy = correct_terms / total_terms if total_terms > 0 else 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_terms': total_terms,
        'correct_terms': correct_terms,
        'per_term_accuracy': term_accuracies,
        'term_stats': term_stats
    }


def evaluate_model(trainer: WhisperTechTrainer, test_data: List[Dict], tech_terms: List[str]) -> Dict[str, Any]:
    """Evaluate model on test data."""
    predictions = []
    references = []
    inference_times = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Evaluating model...", total=len(test_data))
        
        for i, sample in enumerate(test_data):
            try:
                # Get transcription
                import time
                start_time = time.time()
                prediction = trainer.inference(sample['audio_path'])
                inference_time = time.time() - start_time
                
                predictions.append(prediction)
                references.append(sample['transcription'])
                inference_times.append(inference_time)
                
                progress.advance(task)
                
            except Exception as e:
                logger.error(f"Error processing sample {i}: {str(e)}")
                predictions.append("")  # Empty prediction for failed samples
                references.append(sample['transcription'])
                inference_times.append(0.0)
    
    # Calculate general metrics
    general_metrics = calculate_metrics(predictions, references)
    
    # Calculate technical term specific metrics
    tech_metrics = calculate_tech_term_metrics(predictions, references, tech_terms)
    
    # Performance metrics
    avg_inference_time = np.mean(inference_times)
    total_samples = len(test_data)
    
    return {
        'general_metrics': general_metrics,
        'tech_metrics': tech_metrics,
        'performance': {
            'avg_inference_time': avg_inference_time,
            'total_samples': total_samples,
            'successful_predictions': len([p for p in predictions if p])
        },
        'predictions': predictions,
        'references': references
    }


def compare_models(fine_tuned_results: Dict, original_results: Dict) -> Dict[str, Any]:
    """Compare fine-tuned model with original model."""
    comparison = {}
    
    # Compare general metrics
    for metric in fine_tuned_results['general_metrics']:
        ft_value = fine_tuned_results['general_metrics'][metric]
        orig_value = original_results['general_metrics'][metric]
        
        if metric in ['wer', 'cer']:  # Lower is better
            improvement = orig_value - ft_value
            improvement_pct = (improvement / orig_value) * 100 if orig_value > 0 else 0
        else:  # Higher is better
            improvement = ft_value - orig_value
            improvement_pct = (improvement / orig_value) * 100 if orig_value > 0 else 0
        
        comparison[metric] = {
            'fine_tuned': ft_value,
            'original': orig_value,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
    
    # Compare technical term accuracy
    ft_tech_acc = fine_tuned_results['tech_metrics']['overall_accuracy']
    orig_tech_acc = original_results['tech_metrics']['overall_accuracy']
    tech_improvement = ft_tech_acc - orig_tech_acc
    tech_improvement_pct = (tech_improvement / orig_tech_acc) * 100 if orig_tech_acc > 0 else 0
    
    comparison['tech_term_accuracy'] = {
        'fine_tuned': ft_tech_acc,
        'original': orig_tech_acc,
        'improvement': tech_improvement,
        'improvement_pct': tech_improvement_pct
    }
    
    return comparison


def generate_detailed_report(results: Dict, tech_terms: List[str], output_dir: str) -> None:
    """Generate detailed evaluation report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Whisper Fine-tuning Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .good {{ background: #d4edda; }}
            .warning {{ background: #fff3cd; }}
            .error {{ background: #f8d7da; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Whisper Fine-tuning Evaluation Report</h1>
        
        <h2>General Metrics</h2>
        <div class="metric">
            <strong>Word Error Rate (WER):</strong> {results['general_metrics']['wer']:.4f}
        </div>
        <div class="metric">
            <strong>Character Error Rate (CER):</strong> {results['general_metrics']['cer']:.4f}
        </div>
        <div class="metric">
            <strong>BLEU Score:</strong> {results['general_metrics']['bleu']:.4f}
        </div>
        
        <h2>Technical Term Performance</h2>
        <div class="metric">
            <strong>Overall Technical Term Accuracy:</strong> {results['tech_metrics']['overall_accuracy']:.2%}
        </div>
        <div class="metric">
            <strong>Total Technical Terms:</strong> {results['tech_metrics']['total_terms']}
        </div>
        <div class="metric">
            <strong>Correctly Recognized:</strong> {results['tech_metrics']['correct_terms']}
        </div>
        
        <h2>Per-Term Accuracy</h2>
        <table>
            <tr><th>Technical Term</th><th>Accuracy</th><th>Occurrences</th></tr>
    """
    
    for term, accuracy in results['tech_metrics']['per_term_accuracy'].items():
        occurrences = results['tech_metrics']['term_stats'][term]['total']
        css_class = "good" if accuracy > 0.8 else "warning" if accuracy > 0.5 else "error"
        html_content += f'<tr class="{css_class}"><td>{term}</td><td>{accuracy:.2%}</td><td>{occurrences}</td></tr>'
    
    html_content += """
        </table>
        
        <h2>Performance Metrics</h2>
        <div class="metric">
            <strong>Average Inference Time:</strong> {:.3f} seconds
        </div>
        <div class="metric">
            <strong>Total Samples:</strong> {}
        </div>
        <div class="metric">
            <strong>Successful Predictions:</strong> {}
        </div>
        
    </body>
    </html>
    """.format(
        results['performance']['avg_inference_time'],
        results['performance']['total_samples'],
        results['performance']['successful_predictions']
    )
    
    with open(output_path / 'evaluation_report.html', 'w') as f:
        f.write(html_content)
    
    console.print(f"[green]Detailed report saved to: {output_path}/evaluation_report.html[/green]")


def display_results(results: Dict, comparison: Dict = None) -> None:
    """Display evaluation results in a formatted table."""
    
    # General metrics table
    table = Table(title="General Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    if comparison:
        table.add_column("Original", style="yellow")
        table.add_column("Improvement", style="green")
    
    for metric, value in results['general_metrics'].items():
        row = [metric.upper(), f"{value:.4f}"]
        if comparison and metric in comparison:
            row.extend([
                f"{comparison[metric]['original']:.4f}",
                f"{comparison[metric]['improvement']:+.4f} ({comparison[metric]['improvement_pct']:+.1f}%)"
            ])
        table.add_row(*row)
    
    console.print(table)
    
    # Technical terms metrics
    tech_table = Table(title="Technical Terms Performance")
    tech_table.add_column("Metric", style="cyan")
    tech_table.add_column("Value", style="magenta")
    if comparison:
        tech_table.add_column("Original", style="yellow")
        tech_table.add_column("Improvement", style="green")
    
    tech_acc = results['tech_metrics']['overall_accuracy']
    row = ["Technical Term Accuracy", f"{tech_acc:.2%}"]
    if comparison:
        comp_data = comparison['tech_term_accuracy']
        row.extend([
            f"{comp_data['original']:.2%}",
            f"{comp_data['improvement']:+.2%} ({comp_data['improvement_pct']:+.1f}%)"
        ])
    tech_table.add_row(*row)
    
    tech_table.add_row("Total Terms Found", str(results['tech_metrics']['total_terms']))
    tech_table.add_row("Correctly Recognized", str(results['tech_metrics']['correct_terms']))
    
    console.print(tech_table)
    
    # Performance metrics
    perf_table = Table(title="Performance Metrics")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="magenta")
    
    perf = results['performance']
    perf_table.add_row("Average Inference Time", f"{perf['avg_inference_time']:.3f}s")
    perf_table.add_row("Total Samples", str(perf['total_samples']))
    perf_table.add_row("Successful Predictions", str(perf['successful_predictions']))
    
    console.print(perf_table)


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        console.print(f"[red]Model not found: {args.model_path}[/red]")
        console.print("[yellow]Please train a model first with: python scripts/train_model.py[/yellow]")
        sys.exit(1)
    
    # Load technical terms
    terms_file = Path("data/tech_terms.json")
    if not terms_file.exists():
        console.print(f"[red]Technical terms file not found: {terms_file}[/red]")
        sys.exit(1)
    
    tech_terms = load_technical_terms(str(terms_file))
    
    # Determine test data path
    test_data_path = args.test_data or config['data']['val_transcripts']
    if not Path(test_data_path).exists():
        console.print(f"[red]Test data not found: {test_data_path}[/red]")
        console.print("[yellow]Please generate data first with: python scripts/generate_training_data.py[/yellow]")
        sys.exit(1)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Filter for technical terms only if requested
    if args.tech_terms_only:
        test_data = [sample for sample in test_data 
                    if contains_tech_terms(sample['transcription'], tech_terms)]
        console.print(f"[yellow]Evaluating on {len(test_data)} samples containing technical terms[/yellow]")
    
    console.print(Panel(f"Evaluating model on {len(test_data)} samples", style="blue"))
    
    try:
        # Load fine-tuned model
        console.print("Loading fine-tuned model...")
        fine_tuned_trainer = WhisperTechTrainer.load_trained_model(args.model_path, config)
        console.print("[green]✓ Fine-tuned model loaded successfully[/green]")
        
        # Evaluate fine-tuned model
        console.print("\n[bold blue]Evaluating fine-tuned model...[/bold blue]")
        ft_results = evaluate_model(fine_tuned_trainer, test_data, tech_terms)
        
        # Load and evaluate original model if requested
        comparison = None
        if args.compare_original:
            console.print("\n[bold blue]Loading and evaluating original model...[/bold blue]")
            original_config = config.copy()
            original_trainer = WhisperTechTrainer(original_config)
            original_trainer.setup_model()
            
            orig_results = evaluate_model(original_trainer, test_data, tech_terms)
            comparison = compare_models(ft_results, orig_results)
        
        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]EVALUATION RESULTS[/bold green]")
        console.print("="*60)
        
        display_results(ft_results, comparison)
        
        # Generate detailed report if requested
        if args.detailed_report:
            console.print(f"\n[yellow]Generating detailed report...[/yellow]")
            generate_detailed_report(ft_results, tech_terms, args.output_dir)
        
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'evaluation_summary.json', 'w') as f:
            summary = {
                'fine_tuned_results': ft_results,
                'comparison': comparison,
                'config': config,
                'test_samples': len(test_data)
            }
            json.dump(summary, f, indent=2)
        
        console.print(f"\n[green]Results saved to: {output_path}[/green]")
        
        # Summary
        wer = ft_results['general_metrics']['wer']
        tech_acc = ft_results['tech_metrics']['overall_accuracy']
        
        summary_text = f"""[bold]Evaluation Summary:[/bold]
• Word Error Rate: {wer:.2%}
• Technical Term Accuracy: {tech_acc:.2%}
• Test Samples: {len(test_data)}
• Avg Inference Time: {ft_results['performance']['avg_inference_time']:.3f}s"""
        
        if comparison:
            wer_improvement = comparison['wer']['improvement_pct']
            tech_improvement = comparison['tech_term_accuracy']['improvement_pct']
            summary_text += f"""
• WER Improvement: {wer_improvement:+.1f}%
• Technical Term Improvement: {tech_improvement:+.1f}%"""
        
        console.print(Panel(summary_text, title="Summary", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {str(e)}[/red]")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()