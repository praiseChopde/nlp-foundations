"""
HuggingFace NLP Course - Chapter 1
Tokenization, Pipelines, and Transformer Basics

Topics covered:
- Using the pipeline() API for NLP tasks
- Tokenization: converting text to tokens and back
- Understanding input_ids, attention_mask, token_type_ids
- Zero-shot classification
- Text generation
"""

from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# ─────────────────────────────────────────────────────────
# 1. THE PIPELINE API
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("1. PIPELINE API - The easiest way to use a model")
print("=" * 60)

# Sentiment analysis pipeline
sentiment = pipeline("sentiment-analysis")
results = sentiment([
    "I love learning about NLP and language models!",
    "This assignment is too difficult and I'm struggling.",
    "The results were neither good nor bad."
])
for text, result in zip(
    ["Positive sentence", "Negative sentence", "Neutral sentence"],
    results
):
    print(f"  {text}: {result['label']} (confidence: {result['score']:.4f})")

print()

# Zero-shot classification - no training needed
print("Zero-shot classification (no training required):")
classifier = pipeline("zero-shot-classification")
sequence = "This course teaches Python programming and machine learning algorithms."
candidate_labels = ["education", "sports", "politics", "technology", "cooking"]
result = classifier(sequence, candidate_labels)
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label:15s}: {score:.4f}")

print()

# ─────────────────────────────────────────────────────────
# 2. TOKENIZATION - The core of NLP preprocessing
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("2. TOKENIZATION")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Language models are changing how we learn."
print(f"Original text: '{text}'")

# Basic tokenization
tokens = tokenizer.tokenize(text)
print(f"Tokens:        {tokens}")

# Convert tokens to IDs
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs:     {input_ids}")

# Full encoding (adds [CLS] and [SEP] for BERT)
encoding = tokenizer(text, return_tensors="pt")
print(f"\nFull encoding (with special tokens):")
print(f"  input_ids:      {encoding['input_ids'].tolist()}")
print(f"  attention_mask: {encoding['attention_mask'].tolist()}")

# Decoding back to text
decoded = tokenizer.decode(encoding['input_ids'][0])
print(f"\nDecoded back:  '{decoded}'")

print()

# ─────────────────────────────────────────────────────────
# 3. BATCH TOKENIZATION WITH PADDING & TRUNCATION
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("3. BATCH TOKENIZATION (Padding & Truncation)")
print("=" * 60)

sentences = [
    "I want to learn NLP.",
    "Transformers are a type of neural network architecture introduced in the paper Attention is All You Need.",
    "Short sentence."
]

# Without padding - different lengths cause issues
print("Without padding (different lengths):")
for s in sentences:
    enc = tokenizer(s)
    print(f"  '{s[:40]}...' -> length: {len(enc['input_ids'])}")

# With padding and truncation
print("\nWith padding='max_length', truncation=True, max_length=20:")
batch = tokenizer(sentences, padding=True, truncation=True, max_length=20, return_tensors="pt")
print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
print(f"  All sequences same length: {batch['input_ids'].shape[1]}")

print()

# ─────────────────────────────────────────────────────────
# 4. TEXT GENERATION
# ─────────────────────────────────────────────────────────

print("=" * 60)
print("4. TEXT GENERATION with GPT-2")
print("=" * 60)

generator = pipeline("text-generation", model="gpt2", max_new_tokens=40)
prompt = "Educational questions help students learn by"
result = generator(prompt, num_return_sequences=2, truncation=True)
print(f"Prompt: '{prompt}'")
for i, r in enumerate(result):
    print(f"\nGeneration {i+1}:")
    print(f"  {r['generated_text']}")

print()
print("Chapter 1 Complete! Key takeaways:")
print("  - pipeline() abstracts all the complexity for quick NLP tasks")
print("  - Tokenizers convert raw text -> token IDs (numbers the model reads)")
print("  - Padding/truncation ensures uniform sequence lengths for batching")
print("  - Different models use different tokenization strategies")
