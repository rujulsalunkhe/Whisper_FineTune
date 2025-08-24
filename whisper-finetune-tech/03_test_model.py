import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa
from gtts import gTTS
from pydub import AudioSegment
import os

# --- Configuration ---
BASE_MODEL_ID = "openai/whisper-large-v3"
ADAPTER_PATH = "./whisper-large-v3-tech-lora"
TEST_AUDIO_FILE = "test_sample_tech.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def create_test_audio(text, filename):
    """Generates a test audio file from text."""
    print(f"--- Generating test audio: '{text}' ---")
    tts = gTTS(text=text, lang='en')
    tts.save("temp.mp3")
    
    # Convert to 16kHz mono WAV
    audio = AudioSegment.from_mp3("temp.mp3")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(filename, format="wav")
    
    os.remove("temp.mp3")
    print(f"Test audio saved as '{filename}'")

def transcribe(audio_path, model, processor):
    """Transcribes an audio file using the provided model and processor."""
    # Load audio file
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    
    # Process audio
    input_features = processor(
        speech_array, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features.to(DEVICE)
    
    # Generate token ids
    predicted_ids = model.generate(input_features)
    
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    # 1. Create a challenging test audio file
    test_sentence = "First, run mvn clean install, then push to GitHub using git, and finally use the Groq LLM API."
    create_test_audio(test_sentence, TEST_AUDIO_FILE)

    # --- 2. Load and Test the BASE Model ---
    print("\n--- Testing BASE Whisper Model ---")
    base_processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    )

    print("Transcribing with BASE model...")
    base_transcription = transcribe(TEST_AUDIO_FILE, base_model, base_processor)

    # --- 3. Load and Test the FINE-TUNED Model ---
    print("\n--- Testing FINE-TUNED Whisper Model ---")
    # The processor is the same, so we can reuse it or load it again
    finetuned_processor = WhisperProcessor.from_pretrained(ADAPTER_PATH)
    
    # Load the base model again
    base_model_for_finetune = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    )
    
    # Apply the LoRA adapter
    finetuned_model = PeftModel.from_pretrained(base_model_for_finetune, ADAPTER_PATH)

    print("Transcribing with FINE-TUNED model...")
    finetuned_transcription = transcribe(TEST_AUDIO_FILE, finetuned_model, finetuned_processor)

    # --- 4. Compare the Results ---
    print("\n\n" + "="*50)
    print("           PERFORMANCE COMPARISON")
    print("="*50)
    print(f"\nOriginal Sentence:\n  '{test_sentence}'")
    print("-" * 50)
    print(f"\nBase Model Transcription:\n  '{base_transcription}'")
    print("-" * 50)
    print(f"\nFine-Tuned Model Transcription:\n  '{finetuned_transcription}'")
    print("="*50)

    # Clean up the test audio file
    os.remove(TEST_AUDIO_FILE)