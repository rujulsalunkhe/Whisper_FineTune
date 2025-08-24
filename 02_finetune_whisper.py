import torch
from datasets import Dataset, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd
import os
import evaluate
import librosa

# --- Configuration ---
MODEL_ID = "openai/whisper-large-v3"
DATASET_PATH = "./data"
OUTPUT_DIR = "./whisper-large-v3-tech-lora"
METADATA_FILE = os.path.join(DATASET_PATH, "metadata.csv")

# --- 1. Load Processor and Model ---
print("--- Step 1: Loading Processor and Model ---")
# Load the processor which handles feature extraction and tokenization
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="english", task="transcribe")

# Load the model in 8-bit for memory efficiency
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, load_in_8bit=True)

# --- 2. Configure LoRA for Efficient Fine-tuning ---
print("--- Step 2: Configuring LoRA ---")
# Prepare the model for k-bit training (enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
config = LoraConfig(
    r=8,  # Rank of the update matrices. Lower rank means fewer parameters to train.
    lora_alpha=16, # Alpha parameter for scaling.
    target_modules=["q_proj", "v_proj"], # Target modules to apply LoRA to.
    lora_dropout=0.05, # Dropout probability for LoRA layers.
    bias="none" # Do not train bias terms.
)

# Apply LoRA to the model
model = get_peft_model(model, config)
model.print_trainable_parameters() # See how few parameters we are actually training!

# --- 3. Load and Prepare the Dataset ---
print("\n--- Step 3: Loading and Preparing Dataset ---")
# Load metadata into a pandas DataFrame
df = pd.read_csv(METADATA_FILE)
df['file_name'] = df['file_name'].apply(lambda x: os.path.join(DATASET_PATH, x))

# Create a Hugging Face Dataset
raw_dataset = Dataset.from_pandas(df)

# Split the dataset into training and testing sets (90% train, 10% test)
dataset_split = raw_dataset.train_test_split(test_size=0.1)

# Define the data preparation function
def prepare_dataset(batch):
    # compute log-Mel input features from input audio array 
    audio_path = batch["file_name"]
    # Load audio using librosa
    audio_array, sample_rate = librosa.load(audio_path, sr=16000)
    audio_features = processor.feature_extractor(
        audio_array, 
        sampling_rate=16000
    ).input_features[0]
    
    # encode target text to label ids 
    label_ids = processor.tokenizer(batch["transcription"]).input_ids
    return {"input_features": audio_features, "labels": label_ids}

# Apply the preparation function to the dataset
# Note: This is where we are replacing the file paths with actual audio data
# This step might take a few moments
print("Mapping dataset to features...")
tokenized_dataset = dataset_split.map(prepare_dataset, remove_columns=raw_dataset.column_names)

# --- 4. Define Training Configuration ---
print("\n--- Step 4: Defining Training Configuration ---")

# Define a data collator: dynamically pads sequences to the max length in a batch
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Define evaluation metric (Word Error Rate)
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8, # Reduce if you run out of memory
    gradient_accumulation_steps=2, # Effective batch size is batch_size * accumulation_steps
    learning_rate=1e-5,
    warmup_steps=50,
    num_train_epochs=5, # Increase for better performance
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=True, # Use mixed precision for faster training
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

# --- 5. Train the Model ---
print("\n--- Step 5: Initializing Trainer ---")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("--- Starting Training ---")
trainer.train()

# --- 6. Save the Final Model ---
print("\n--- Step 6: Saving the Fine-tuned Model ---")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print("\n--- Fine-tuning Complete! ---")
print(f"Model adapters and processor saved to '{OUTPUT_DIR}'")