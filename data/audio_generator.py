"""
Audio generation module for creating training data with technical terms.
"""

import json
import random
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import edge_tts
from gtts import gTTS
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechTermAudioGenerator:
    """Generate audio samples for technical terms using various TTS engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.terms_data = None
        self.voices = config.get('voices', ['en-US-AriaNeural'])
        self.sample_rate = config.get('sample_rate', 16000)
        self.samples_per_term = config.get('samples_per_term', 20)
        
    def load_technical_terms(self, terms_file: str) -> None:
        """Load technical terms from JSON file."""
        with open(terms_file, 'r') as f:
            self.terms_data = json.load(f)
            
    async def generate_with_edge_tts(self, text: str, voice: str, output_path: str) -> bool:
        """Generate audio using Edge TTS."""
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            return True
        except Exception as e:
            logger.error(f"Edge TTS generation failed: {e}")
            return False
            
    def generate_with_gtts(self, text: str, output_path: str) -> bool:
        """Generate audio using Google TTS."""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            return True
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return False
            
    def add_background_noise(self, audio: AudioSegment, noise_level: float = 0.1) -> AudioSegment:
        """Add subtle background noise to make audio more realistic."""
        # Generate white noise
        duration = len(audio)
        noise_samples = np.random.normal(0, 1, int(duration * audio.frame_rate / 1000))
        noise_audio = AudioSegment(
            noise_samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=1
        )
        
        # Mix with original audio
        noise_audio = noise_audio - (60 - int(noise_level * 60))  # Reduce noise volume
        return audio.overlay(noise_audio)
        
    def apply_speed_variation(self, audio: AudioSegment, speed_range: Tuple[float, float] = (0.9, 1.1)) -> AudioSegment:
        """Apply random speed variation to audio."""
        speed_factor = random.uniform(*speed_range)
        new_frame_rate = int(audio.frame_rate * speed_factor)
        return audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
        
    def normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Normalize audio levels."""
        return normalize(audio)
        
    def create_sentence_variations(self, term: str, templates: List[str]) -> List[str]:
        """Create sentence variations using the technical term."""
        sentences = []
        
        # Use provided templates
        for template in templates:
            sentence = template.format(term=term)
            sentences.append(sentence)
            
        # Add some natural variations
        natural_sentences = [
            f"We're implementing {term} in our new system.",
            f"Have you tried using {term} for this use case?",
            f"The {term} integration went smoothly.",
            f"I recommend checking out {term} for your project.",
            f"Our {term} setup is working perfectly now.",
            f"Let me show you how {term} works.",
            f"The {term} configuration needs to be updated.",
            f"We've been using {term} for the past six months.",
            f"Can you walk me through the {term} implementation?",
            f"The {term} performance metrics look great."
        ]
        
        sentences.extend(random.sample(natural_sentences, min(5, len(natural_sentences))))
        return sentences
        
    async def generate_audio_samples(self, output_dir: str, train_split: float = 0.8) -> Dict[str, List[Dict]]:
        """Generate audio samples for all technical terms."""
        output_path = Path(output_dir)
        train_dir = output_path / "train_audio"
        val_dir = output_path / "val_audio"
        
        # Create directories
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        train_data = []
        val_data = []
        
        templates = self.config.get('templates', [
            "I'm using {term} for this project",
            "Let's configure {term} properly"
        ])
        
        # Process each category of technical terms
        for category, terms in self.terms_data['technical_terms'].items():
            if category == 'sentence_contexts' or category == 'background_contexts':
                continue
                
            logger.info(f"Processing category: {category}")
            
            for term_data in terms:
                term = term_data['term']
                variations = term_data.get('variations', [term])
                
                logger.info(f"Generating audio for term: {term}")
                
                # Generate sentences for this term and its variations
                all_sentences = []
                for variation in variations:
                    sentences = self.create_sentence_variations(variation, templates)
                    all_sentences.extend(sentences)
                
                # Generate audio samples
                sample_count = 0
                for sentence in all_sentences[:self.samples_per_term]:
                    for voice in self.voices:
                        if sample_count >= self.samples_per_term:
                            break
                            
                        # Determine train/val split
                        is_train = random.random() < train_split
                        target_dir = train_dir if is_train else val_dir
                        target_list = train_data if is_train else val_data
                        
                        # Generate filename
                        filename = f"{term}_{sample_count}_{voice.replace('-', '_')}.wav"
                        file_path = target_dir / filename
                        
                        # Generate audio
                        temp_path = str(file_path).replace('.wav', '_temp.wav')
                        success = await self.generate_with_edge_tts(sentence, voice, temp_path)
                        
                        if success:
                            # Process audio
                            audio = AudioSegment.from_wav(temp_path)
                            
                            # Apply augmentations
                            if self.config.get('apply_noise', False) and random.random() < 0.3:
                                audio = self.add_background_noise(audio, 0.1)
                                
                            if self.config.get('speed_perturbation', False):
                                audio = self.apply_speed_variation(audio)
                                
                            # Normalize
                            audio = self.normalize_audio(audio)
                            
                            # Convert to target sample rate
                            audio = audio.set_frame_rate(self.sample_rate)
                            audio = audio.set_channels(1)  # Mono
                            
                            # Save final audio
                            audio.export(str(file_path), format="wav")
                            
                            # Add to dataset
                            target_list.append({
                                "audio_path": str(file_path),
                                "transcription": sentence,
                                "term": term,
                                "voice": voice,
                                "category": category
                            })
                            
                            sample_count += 1
                            
                            # Clean up temp file
                            Path(temp_path).unlink(missing_ok=True)
                        else:
                            logger.warning(f"Failed to generate audio for: {sentence}")
                            
                logger.info(f"Generated {sample_count} samples for {term}")
        
        return {
            "train": train_data,
            "validation": val_data
        }
        
    def save_dataset_metadata(self, dataset: Dict[str, List[Dict]], output_dir: str) -> None:
        """Save dataset metadata to JSON files."""
        output_path = Path(output_dir)
        
        # Save train metadata
        with open(output_path / "train_transcripts.json", 'w') as f:
            json.dump(dataset['train'], f, indent=2)
            
        # Save validation metadata
        with open(output_path / "val_transcripts.json", 'w') as f:
            json.dump(dataset['validation'], f, indent=2)
            
        logger.info(f"Saved metadata for {len(dataset['train'])} train and {len(dataset['validation'])} validation samples")


async def main():
    """Main function to generate training data."""
    config = {
        'voices': [
            'en-US-AriaNeural',
            'en-US-JennyNeural', 
            'en-US-GuyNeural',
            'en-GB-SoniaNeural'
        ],
        'sample_rate': 16000,
        'samples_per_term': 15,
        'apply_noise': True,
        'speed_perturbation': True,
        'templates': [
            "I'm using {term} for this project",
            "Let's install {term} first",
            "The {term} documentation is helpful",
            "Can you help me with {term}?",
            "I need to configure {term} properly",
            "The {term} API is quite powerful",
            "We should migrate to {term}",
            "How do I use {term} effectively?"
        ]
    }
    
    generator = TechTermAudioGenerator(config)
    generator.load_technical_terms('tech_terms.json')
    
    logger.info("Starting audio generation...")
    dataset = await generator.generate_audio_samples('../data')
    generator.save_dataset_metadata(dataset, '../data')
    
    logger.info("Audio generation completed!")
    logger.info(f"Generated {len(dataset['train'])} training samples")
    logger.info(f"Generated {len(dataset['validation'])} validation samples")


if __name__ == "__main__":
    asyncio.run(main())