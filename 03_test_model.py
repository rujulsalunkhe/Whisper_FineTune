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
TEST_AUDIO_FILE = "my_audio.wav"

USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if USE_CUDA else "cpu"
# Use fp16 on CUDA; fp32 otherwise
MODEL_DTYPE = torch.float16 if USE_CUDA else torch.float32

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

def transcribe(audio_path, model: WhisperForConditionalGeneration, processor: WhisperProcessor):
    """Transcribes an audio file using the provided model and processor."""
    # Load audio at 16kHz
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

    # Prepare input features (float32 by default) then cast to model dtype/device
    inputs = processor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt"
    )
    input_features = inputs.input_features.to(device=DEVICE, dtype=model.dtype)

    # Ensure generation config for language/task to avoid warnings and improve stability
    # Newer Transformers support setting these on generation_config:
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"

    # Older-safe path: pass forced_decoder_ids directly
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )

    # Decode to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def preprocess_audio(input_path, output_path="processed.wav"):
    """Ensure audio is 16kHz mono WAV for Whisper compatibility."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path


if __name__ == "__main__":
    # 1. Create a challenging test audio file
    # test_sentence = "First, run mvn clean install, then push to GitHub using git, and finally use the Groq LLM API."
    # create_test_audio(test_sentence, TEST_AUDIO_FILE)

    # --- 2. Load and Test the BASE Model ---
    print("\n--- Testing BASE Whisper Model ---")
    base_processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=MODEL_DTYPE
    ).to(DEVICE)

    print("Transcribing with BASE model...")
    base_transcription = transcribe(TEST_AUDIO_FILE, base_model, base_processor)

    # --- 3. Load and Test the FINE-TUNED Model (LoRA) ---
    print("\n--- Testing FINE-TUNED Whisper Model ---")
    finetuned_processor = WhisperProcessor.from_pretrained(ADAPTER_PATH)  # ok to reuse base_processor too

    base_model_for_finetune = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=MODEL_DTYPE
    ).to(DEVICE)

    finetuned_model = PeftModel.from_pretrained(base_model_for_finetune, ADAPTER_PATH)
    # Optional: merge LoRA into the base weights for slightly faster inference
    # finetuned_model = finetuned_model.merge_and_unload()

    print("Transcribing with FINE-TUNED model...")
    finetuned_transcription = transcribe(TEST_AUDIO_FILE, finetuned_model, finetuned_processor)

    # --- 4. Compare the Results ---
    print("\n\n" + "="*50)
    print("           PERFORMANCE COMPARISON")
    print("="*50)
    print(f"\nAudio File Tested:\n  '{TEST_AUDIO_FILE}'")
    print("-" * 50)
    # print(f"\nBase Model Transcription:\n  '{base_transcription}'")
    # print("-" * 50)
    print(f"\nFine-Tuned Model Transcription:\n  '{finetuned_transcription}'")
    print("="*50)

    # Clean up the test audio file
    # os.remove(TEST_AUDIO_FILE)
