"""
HuggingFace NLP Course - Chapter 3
Full Fine-tuning Workflow with Datasets and Evaluate

Topics covered:
- Loading and exploring datasets with datasets library
- Preprocessing with map()
- Dynamic padding and DataCollators
- Training loop WITHOUT Trainer (manual PyTorch)
- Training loop WITH Trainer
- Evaluation with evaluate library
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler,
)
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────
# 1. LOADING AND EXPLORING A DATASET
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("1. LOADING AND EXPLORING DATASETS")
print("=" * 60)

# MRPC: Microsoft Research Paraphrase Corpus
# Task: determine if two sentences are paraphrases
raw_datasets = load_dataset("glue", "mrpc")

print(f"Dataset splits: {list(raw_datasets.keys())}")
print(f"Training examples: {len(raw_datasets['train'])}")
print(f"Validation examples: {len(raw_datasets['validation'])}")
print(f"\nColumn names: {raw_datasets['train'].column_names}")
print(f"\nExample:")
example = raw_datasets["train"][0]
print(f"  Sentence 1: {example['sentence1']}")
print(f"  Sentence 2: {example['sentence2']}")
print(f"  Label (1=paraphrase): {example['label']}")

print()

# ─────────────────────────────────────────────────────────
# 2. PREPROCESSING WITH MAP()
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("2. PREPROCESSING WITH map()")
print("=" * 60)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True,
    )

# Apply tokenization to the entire dataset at once
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"Columns after tokenization: {tokenized_datasets['train'].column_names}")

# Remove columns the model doesn't need
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print(f"Columns after cleanup: {tokenized_datasets['train'].column_names}")
print(f"Format: {tokenized_datasets['train'].format['type']}")

print()

# ─────────────────────────────────────────────────────────
# 3. DYNAMIC PADDING WITH DataCollator
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("3. DYNAMIC PADDING WITH DataCollatorWithPadding")
print("=" * 60)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Compare fixed-length padding vs dynamic padding
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["labels"]}
lengths = [len(x) for x in samples["input_ids"]]
print(f"Sequence lengths before padding: {lengths}")

batch = data_collator(
    [{k: v[i] for k, v in samples.items()} for i in range(len(samples["input_ids"]))]
)
print(f"Batch input_ids shape after dynamic padding: {batch['input_ids'].shape}")
print("(Padded to the longest sequence in the batch, not a fixed max length)")

print()

# ─────────────────────────────────────────────────────────
# 4. MANUAL PYTORCH TRAINING LOOP
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("4. MANUAL PYTORCH TRAINING LOOP")
print("=" * 60)

# Use a small subset for speed
small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
small_eval = tokenized_datasets["validation"].select(range(100))

train_dataloader = DataLoader(
    small_train, shuffle=True, batch_size=16, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    small_eval, batch_size=16, collate_fn=data_collator
)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Learning rate scheduler
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on: {device}")

# Training loop
model.train()
print(f"Training for {num_training_steps} steps...")
for batch in tqdm(train_dataloader, desc="Training"):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

# Evaluation loop
metric = evaluate.load("glue", "mrpc")
model.eval()
print("Evaluating...")
for batch in tqdm(eval_dataloader, desc="Evaluating"):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

results = metric.compute()
print(f"\nResults after 1 epoch (200 training examples):")
print(f"  Accuracy: {results['accuracy']:.4f}")
print(f"  F1 Score: {results['f1']:.4f}")

print()
print("Chapter 3 Complete! Key takeaways:")
print("  - datasets.map() applies preprocessing efficiently to large datasets")
print("  - DataCollatorWithPadding pads each batch to its longest sequence (efficient)")
print("  - Manual training loop: forward pass -> loss -> backward -> optimizer step")
print("  - Always use model.train() for training and model.eval() for evaluation")
print("  - evaluate library provides standardized metrics for NLP benchmarks")
