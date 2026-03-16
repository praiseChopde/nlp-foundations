"""
HuggingFace NLP Course - Chapter 2
Behind the Pipeline: Transformer Architecture and Fine-tuning

Topics covered:
- What happens inside pipeline()
- AutoModel, AutoTokenizer
- Encoders vs Decoders vs Encoder-Decoders
- Fine-tuning with Trainer API
- Training loop from scratch
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np

# ─────────────────────────────────────────────────────────
# 1. WHAT HAPPENS INSIDE pipeline()
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("1. INSIDE THE PIPELINE - Manual steps")
print("=" * 60)

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "Retrieval-Augmented Generation improves answer quality.",
    "This model produces irrelevant and unhelpful outputs.",
]

# Step 1: Tokenize
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(f"Step 1 - Tokenized inputs shape: {inputs['input_ids'].shape}")

# Step 2: Forward pass through model
with torch.no_grad():
    outputs = model(**inputs)
print(f"Step 2 - Raw logits (model output):\n  {outputs.logits}")

# Step 3: Convert logits -> probabilities -> labels
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
predictions = torch.argmax(probabilities, dim=-1)
labels = [model.config.id2label[pred.item()] for pred in predictions]

print(f"Step 3 - Probabilities: {probabilities.tolist()}")
print(f"Step 3 - Predicted labels: {labels}")
print()

# ─────────────────────────────────────────────────────────
# 2. ENCODER vs DECODER vs ENCODER-DECODER
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("2. TRANSFORMER ARCHITECTURE TYPES")
print("=" * 60)

architecture_table = {
    "Encoder-only (BERT, DistilBERT)": {
        "use_case": "Classification, NER, extractive QA",
        "bidirectional": True,
        "example_task": "Sentence classification, feature extraction"
    },
    "Decoder-only (GPT-2, LLaMA)": {
        "use_case": "Text generation, causal language modeling",
        "bidirectional": False,
        "example_task": "Question generation, story completion"
    },
    "Encoder-Decoder (T5, BART)": {
        "use_case": "Translation, summarization, question generation",
        "bidirectional": "Both",
        "example_task": "Seq2seq tasks like educational Q generation"
    },
}

for arch, details in architecture_table.items():
    print(f"\n  {arch}")
    print(f"    Use case:        {details['use_case']}")
    print(f"    Bidirectional:   {details['bidirectional']}")
    print(f"    Example task:    {details['example_task']}")

print()

# ─────────────────────────────────────────────────────────
# 3. EXTRACTING HIDDEN STATES (Embeddings)
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("3. EXTRACTING EMBEDDINGS FROM A MODEL")
print("=" * 60)

tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased")

sentences = [
    "Students learn better with good questions.",
    "Effective pedagogy requires meaningful feedback.",
    "The weather today is sunny and warm.",
]

with torch.no_grad():
    encoded = tokenizer_bert(sentences, padding=True, truncation=True, return_tensors="pt")
    outputs = model_bert(**encoded)

# CLS token embedding = sentence representation
cls_embeddings = outputs.last_hidden_state[:, 0, :]
print(f"Embedding shape per sentence: {cls_embeddings.shape}")

# Cosine similarity to show semantic meaning is captured
from torch.nn.functional import cosine_similarity
sim_01 = cosine_similarity(cls_embeddings[0].unsqueeze(0), cls_embeddings[1].unsqueeze(0))
sim_02 = cosine_similarity(cls_embeddings[0].unsqueeze(0), cls_embeddings[2].unsqueeze(0))
print(f"\nCosine similarity:")
print(f"  'Students learn...' vs 'Effective pedagogy...': {sim_01.item():.4f} (semantically related)")
print(f"  'Students learn...' vs 'The weather today...': {sim_02.item():.4f} (unrelated)")

print()

# ─────────────────────────────────────────────────────────
# 4. FINE-TUNING WITH TRAINER API
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("4. FINE-TUNING - Trainer API Example")
print("=" * 60)

# Load a small dataset for demonstration
dataset = load_dataset("glue", "sst2")
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Only use a small slice for demonstration
small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval = tokenized_datasets["validation"].shuffle(seed=42).select(range(50))

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    report_to="none",
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning on small SST-2 subset (100 examples, 1 epoch)...")
print("This demonstrates the complete fine-tuning loop.\n")
trainer.train()

results = trainer.evaluate()
print(f"\nFine-tuning complete!")
print(f"Evaluation accuracy: {results['eval_accuracy']:.4f}")
print()

print("Chapter 2 Complete! Key takeaways:")
print("  - pipeline() = tokenize + model forward pass + postprocess")
print("  - Encoder models: best for classification/understanding tasks")
print("  - Decoder models: best for text generation")
print("  - Encoder-decoder: best for seq2seq (e.g. question generation)")
print("  - Fine-tuning = updating a pretrained model on your specific task")
print("  - Trainer API handles the training loop, evaluation, and checkpointing")
