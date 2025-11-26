"""
train_miniLM_faiss.py
Mid-sized MiniLM intent classifier + FAISS fallback.

Usage:
    python train_miniLM_faiss.py

Files expected (local):
- /mnt/data/unified_dataset.json   (preferred)
- /mnt/data/chattea.csv            (fallback)
- /mnt/data/responses_bilingual.json

Outputs (saved under artifacts/):
- artifacts/intent_model/          (HuggingFace model + tokenizer)
- artifacts/faiss.index            (FAISS index file)
- artifacts/embeddings.npy         (embeddings matrix)
- artifacts/texts_intents.json     (index -> text/intent mapping)
- artifacts/label_mappings.json
"""

import os
import json
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sentence_transformers import SentenceTransformer
import logging

# Optional: langdetect
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# Optional: FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ----------------------------------------
# CONFIG
# ----------------------------------------
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
# Prefer unified dataset if present, otherwise fallback to uploaded CSV
UNIFIED_PATH = Path("/mnt/data/unified_dataset.json")
CSV_PATH = Path("/mnt/data/chattea.csv")
BILINGUAL_RESP_PATH = Path("/mnt/data/responses_bilingual.json")

ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_MODEL_DIR = ARTIFACTS / "intent_model"
EMBEDDINGS_PATH = ARTIFACTS / "embeddings.npy"
FAISS_INDEX_PATH = ARTIFACTS / "faiss.index"
TEXTS_INTENTS_PATH = ARTIFACTS / "texts_intents.json"
LABEL_MAP_PATH = ARTIFACTS / "label_mappings.json"

BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 128
FP16 = True  # use mixed precision if GPU available
CONFIDENCE_THRESHOLD = 0.6  # if classifier confidence < this, fallback to FAISS
TOP_K_FAISS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {DEVICE} | FAISS available: {FAISS_AVAILABLE} | langdetect: {LANGDETECT_AVAILABLE}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chattea")

# ----------------------------------------
# UTILITIES
# ----------------------------------------
def load_unified_or_csv():
    """
    Load dataset from unified json (preferred) or fallback to csv.
    Expects fields: text, intent
    """
    if UNIFIED_PATH.exists():
        print(f"Loading unified dataset: {UNIFIED_PATH}")
        with open(UNIFIED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame([{"text": item["text"], "intent": item["intent"]} for item in data])
    elif CSV_PATH.exists():
        print(f"Loading CSV dataset: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        # Expect columns 'text' and 'intent'
        if "text" not in df.columns or "intent" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'intent' columns")
        df = df[["text", "intent"]].copy()
    else:
        raise FileNotFoundError("No dataset file found. Place unified_dataset.json or chattea.csv in /mnt/data/")

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"Loaded {len(df)} samples, unique intents: {df['intent'].nunique()}")
    return df

def load_bilingual_responses():
    if not BILINGUAL_RESP_PATH.exists():
        raise FileNotFoundError(f"Bilingual responses file missing: {BILINGUAL_RESP_PATH}")
    with open(BILINGUAL_RESP_PATH, "r", encoding="utf-8") as f:
        responses_bilingual = json.load(f)
    print(f"Loaded bilingual responses for {len(responses_bilingual)} intents")
    return responses_bilingual

def preprocess_text(text):
    t = str(text).lower()
    t = " ".join(t.split())
    return t

def detect_language_simple(text):
    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            if lang and (lang.startswith("id") or lang == "ms"):
                return "id"
            else:
                return "en"
        except Exception:
            pass
    # simple heuristic fallback
    id_words = {"cara","kirim","pesan","jadwal","nomor","cek","bantuan","filter","panasin","harga"}
    tokens = text.lower().split()
    if len(tokens) == 0:
        return "en"
    id_count = sum(1 for t in tokens if t in id_words)
    if id_count >= max(1, int(0.3 * len(tokens))):
        return "id"
    return "en"

# ----------------------------------------
# PREPARE DATA
# ----------------------------------------
df = load_unified_or_csv()
responses_bilingual = load_bilingual_responses()

# Build label mappings
unique_intents = sorted(df["intent"].unique())
label2id = {label: idx for idx, label in enumerate(unique_intents)}
id2label = {idx: label for label, idx in label2id.items()}

df["label"] = df["intent"].map(label2id)
print("Label mapping sample:", list(label2id.items())[:10])

# Save mapping
with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)

# Train/val/test split (stratified)
train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
train_df, val_df = train_test_split(train_val_df, test_size=0.10, stratify=train_val_df["label"], random_state=SEED)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Create HF DatasetDict
train_ds = Dataset.from_pandas(train_df[["text", "label"]])
val_ds = Dataset.from_pandas(val_df[["text", "label"]])
test_ds = Dataset.from_pandas(test_df[["text", "label"]])
dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

# ----------------------------------------
# TOKENIZER & MODEL
# ----------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

print("Tokenizing datasets...")
tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------------------
# TRAINING
# ----------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    pr, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": pr, "recall": rec, "f1": f1}

training_args = TrainingArguments(
    output_dir=str(OUTPUT_MODEL_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=FP16 and DEVICE == "cuda",
    logging_steps=50,
    save_total_limit=3,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
trainer.save_model(str(OUTPUT_MODEL_DIR))
tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))
print("Model and tokenizer saved to:", OUTPUT_MODEL_DIR)

# Evaluate on test
print("Evaluating on test set...")
preds_output = trainer.predict(tokenized["test"])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)
print("\nClassification report on TEST:")
print(classification_report(y_true, y_pred, target_names=unique_intents))

# ----------------------------------------
# BUILD FAISS INDEX (sentence-transformers) from all texts (train+val+test)
# ----------------------------------------
print("\nBuilding sentence-transformers embeddings and FAISS index...")
embedder = SentenceTransformer(MODEL_NAME)

all_texts = (train_df["text"].tolist() + val_df["text"].tolist() + test_df["text"].tolist())
all_intents = (train_df["intent"].tolist() + val_df["intent"].tolist() + test_df["intent"].tolist())

# Preprocess texts consistently
all_texts_proc = [preprocess_text(t) for t in all_texts]

embeddings = embedder.encode(all_texts_proc, convert_to_numpy=True, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)
np.save(EMBEDDINGS_PATH, embeddings)
# Save text-intent mapping
index_to_item = [{"text": t, "intent": it} for t, it in zip(all_texts_proc, all_intents)]
with open(TEXTS_INTENTS_PATH, "w", encoding="utf-8") as f:
    json.dump(index_to_item, f, indent=2, ensure_ascii=False)

# Build FAISS index (L2 normalized inner product is fine for cosine)
d = embeddings.shape[1]
print("FAISS dimension:", d)
# normalize embeddings for cosine
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1e-10
embeddings_norm = embeddings / norms

if FAISS_AVAILABLE:
    # use IndexFlatIP on normalized vectors -> cosine similarity
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_norm.astype('float32'))
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print("FAISS index saved to:", FAISS_INDEX_PATH)
else:
    index = None
    print("FAISS not available. Install faiss-cpu or faiss-gpu to enable semantic fallback.")

# ----------------------------------------
# INFERENCE: classifier + FAISS fallback
# ----------------------------------------
# reload label mappings into model config if needed
# Ensure model.config.id2label is present (Trainer set it earlier)
print("Model config id2label sample:", list(model.config.id2label.items())[:5])

def classifier_predict_intent(text):
    """
    Returns (intent_key, confidence)
    confidence = softmax probability of predicted class
    """
    text_pre = preprocess_text(text)
    inputs = tokenizer(text_pre, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        intent_key = model.config.id2label[pred_idx]
    return intent_key, confidence

def faiss_fallback(text, top_k=TOP_K_FAISS):
    """
    Returns the intent of the top retrieved item from FAISS (cosine)
    """
    if index is None:
        return None, 0.0
    q = preprocess_text(text)
    q_emb = embedder.encode([q], convert_to_numpy=True)[0]
    qn = q_emb / (np.linalg.norm(q_emb) + 1e-10)
    qn = qn.astype('float32').reshape(1, -1)
    D, I = index.search(qn, top_k)  # inner product scores, indices
    # take top result
    best_idx = int(I[0][0])
    score = float(D[0][0])
    mapped = index_to_item[best_idx]
    return mapped["intent"], float(score)

def get_bilingual_response(intent_key, user_lang):
    """
    Return response string for intent and language with fallbacks.
    responses_bilingual: dict(intent -> {"id": "...", "en": "..."})
    """
    entry = responses_bilingual.get(intent_key, {})
    if not isinstance(entry, dict):
        # in case responses file stores direct strings for some intents
        return entry if entry else "Maaf, saya belum memiliki jawaban."
    # normalize keys
    entry_norm = {k.lower(): v for k, v in entry.items()}
    resp = entry_norm.get(user_lang)
    if resp:
        return resp
    return entry_norm.get("id") or entry_norm.get("en") or "Maaf, saya belum memiliki jawaban."

def predict_and_respond(text, confidence_threshold=CONFIDENCE_THRESHOLD):
    # step 1: classifier
    intent_cls, conf = classifier_predict_intent(text)
    chosen_intent = intent_cls
    source = "classifier"
    # if low confidence, try FAISS
    if conf < confidence_threshold and FAISS_AVAILABLE:
        intent_faiss, score = faiss_fallback(text, top_k=TOP_K_FAISS)
        if intent_faiss is not None:
            # we can compare raw scores (cosine) vs classifier confidence scaled — simple rule:
            # if faiss returns something, use it (or apply a small acceptance rule)
            chosen_intent = intent_faiss
            source = "faiss"
    # choose language and response
    lang = detect_language_simple(text)
    response_text = get_bilingual_response(chosen_intent, lang)
    return {"intent": chosen_intent, "confidence": conf, "source": source, "response": response_text}

# ----------------------------------------
# DEMO
# ----------------------------------------
print("\n--- DEMO ---")
demo_queries = [
    "gimana cara blast message?",
    "how to schedule messages?",
    "cek nomor wa yang valid",
    "tolong bantu saya",
    "what is the pricing plan?"
]

for q in demo_queries:
    res = predict_and_respond(q)
    print(f"\nUser: {q}")
    print(f"-> intent: {res['intent']} (conf={res['confidence']:.3f}) [source={res['source']}]")
    print(f"-> response: {res['response']}")

print("\nScript finished. Artifacts saved to:", ARTIFACTS)
