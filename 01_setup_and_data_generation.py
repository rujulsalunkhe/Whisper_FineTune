import os
import csv
from gtts import gTTS
from pydub import AudioSegment
import random
import shutil

# --- Configuration ---
# Your target technical terms
TECHNICAL_TERMS = [
    "mvn", "maven", "github", "git", "portkey", "openai", 
    "chatgpt", "llm", "groq", "Grok"
]

# Simple sentence templates to provide context
SENTENCE_TEMPLATES = [
    "First, you need to run {}.",
    "Did you check the documentation for {}?",
    "Let's talk about the {} API.",
    "My latest project involves using {}.",
    "I pushed the code to {}.",
    "How does {} work internally?",
    "We should integrate {} into our workflow.",
    "The performance of {} is impressive.",
    "Let's troubleshoot the issue with {}.",
    "Can you explain {} to me?",
]

# Dataset parameters
NUM_SAMPLES_PER_TERM = 10  # Increase for a more robust dataset
OUTPUT_DIR = "data"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")
TARGET_SAMPLE_RATE = 16000 # Whisper's required sample rate

def create_synthetic_audio_dataset():
    """
    Generates a synthetic audio dataset for fine-tuning Whisper.
    Creates audio files from text and a metadata file for the dataset.
    """
    print("--- Starting Synthetic Data Generation ---")

    # 1. Setup directory structure
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' already exists. Deleting it.")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(AUDIO_DIR, exist_ok=True)
    print(f"Created directories: {OUTPUT_DIR} and {AUDIO_DIR}")

    metadata = []
    sample_id = 1

    # 2. Generate audio for each term
    for term in TECHNICAL_TERMS:
        print(f"\nProcessing term: '{term}'")
        for i in range(NUM_SAMPLES_PER_TERM):
            # Create a sentence with the term
            sentence = random.choice(SENTENCE_TEMPLATES).format(term)
            
            # Use a unique filename
            filename_mp3 = f"sample_{sample_id:04d}.mp3"
            filepath_mp3 = os.path.join(AUDIO_DIR, filename_mp3)
            
            # Generate speech using gTTS
            try:
                tts = gTTS(text=sentence, lang='en', slow=False)
                tts.save(filepath_mp3)
            except Exception as e:
                print(f"  - Could not generate audio for '{sentence}'. Error: {e}")
                continue

            # Convert mp3 to 16kHz mono WAV (Whisper's required format)
            filename_wav = f"sample_{sample_id:04d}.wav"
            filepath_wav = os.path.join(AUDIO_DIR, filename_wav)
            
            try:
                audio = AudioSegment.from_mp3(filepath_mp3)
                audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
                audio.export(filepath_wav, format="wav")
                os.remove(filepath_mp3) # Clean up the mp3 file
                
                # Add to metadata
                # We need the relative path for the dataset loader
                relative_path = os.path.join("audio", filename_wav)
                metadata.append([relative_path, sentence])
                print(f"  - Generated: {filename_wav} | Transcript: '{sentence}'")
                sample_id += 1

            except Exception as e:
                print(f"  - Failed to process {filepath_mp3}. Error: {e}")
                if os.path.exists(filepath_mp3):
                    os.remove(filepath_mp3)
    
    # 3. Write metadata to CSV
    with open(METADATA_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'transcription']) # Header
        writer.writerows(metadata)

    print(f"\n--- Data Generation Complete ---")
    print(f"Total samples created: {len(metadata)}")
    print(f"Dataset ready in '{OUTPUT_DIR}' directory.")
    print(f"Metadata file created at '{METADATA_FILE}'")


if __name__ == "__main__":
    create_synthetic_audio_dataset()