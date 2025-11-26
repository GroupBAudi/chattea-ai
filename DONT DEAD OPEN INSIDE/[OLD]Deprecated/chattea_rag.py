"""
chattea_rag_hybrid_final.py
Hybrid TF-IDF + Sentence-Transformer RAG intent matcher (unified dataset)
with bilingual response selection (ID/EN) based on detected user language.

Defaults expect:
 - /mnt/data/unified_dataset.json             (or unified_dataset.json next to script)
 - /mnt/data/responses_bilingual.json         (or responses_bilingual.json next to script)

Install (optional for best results):
    pip install sentence-transformers langdetect scipy

Usage:
    python chattea_rag_hybrid_final.py
"""

import os
import re
import json
import pickle
import numpy as np
from pathlib import Path
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn import metrics

# ---------------------------
# Paths (auto-adapt to environment)
# ---------------------------
# Prefer sandbox path if present, otherwise use local script dir.
SANDBOX_DIR = Path("/mnt/data")
BASE_DIR = SANDBOX_DIR if SANDBOX_DIR.exists() else Path(__file__).parent

UNIFIED_PATH = BASE_DIR / "dataset.json"
BILINGUAL_RESP_PATH = BASE_DIR / "responses_bilingual.json"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.npz"
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
TEXTS_INTENTS_PATH = ARTIFACTS_DIR / "texts_intents.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Load unified dataset (texts + intent)
# ---------------------------
if not UNIFIED_PATH.exists():
    raise FileNotFoundError(f"Unified dataset not found at {UNIFIED_PATH}")

with open(UNIFIED_PATH, "r", encoding="utf-8") as f:
    unified = json.load(f)

# Use preprocessed texts for vectorizers
texts = [preprocess(item["text"]) for item in unified]
intents = [item["intent"] for item in unified]
# Note: unified might contain a single-language response field; we will use bilingual mapping below.
print(f"Loaded unified dataset: {len(texts)} samples from {UNIFIED_PATH}")

# Save mapping for inspection
index_to_item = [{"text": t, "intent": it} for t, it in zip(texts, intents)]
with open(TEXTS_INTENTS_PATH, "w", encoding="utf-8") as f:
    json.dump(index_to_item, f, indent=2, ensure_ascii=False)
print(f"Saved texts/intents mapping to {TEXTS_INTENTS_PATH}")

# ---------------------------
# Load bilingual responses
# ---------------------------
if not BILINGUAL_RESP_PATH.exists():
    raise FileNotFoundError(f"Bilingual responses file not found at {BILINGUAL_RESP_PATH}")

with open(BILINGUAL_RESP_PATH, "r", encoding="utf-8") as f:
    responses_bilingual = json.load(f)

# Expect structure: { "intent_key": {"id": "...", "en": "..."}, ... }
print(f"Loaded bilingual responses (intents): {len(responses_bilingual)}")

# ---------------------------
# TF-IDF Vectorizer
# ---------------------------
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# save vectorizer
with open(TFIDF_VECTORIZER_PATH, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"Saved TF-IDF vectorizer to {TFIDF_VECTORIZER_PATH}")

# optionally save sparse matrix
try:
    from scipy import sparse
    sparse.save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
    print(f"Saved TF-IDF matrix to {TFIDF_MATRIX_PATH}")
except Exception:
    pass

# ---------------------------
# Embeddings (optional)
# ---------------------------
embedding_available = False
embeddings = None
model = None

try:
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    embedding_available = True
    print(f"Saved embeddings to {EMBEDDINGS_PATH} (shape={embeddings.shape})")
except Exception as e:
    print("SentenceTransformers unavailable or failed to load. Running TF-IDF only.")
    # print(e)  # optional debug
    embedding_available = False

# ---------------------------
# Language detection helper
# ---------------------------
try:
    from langdetect import detect
    langdetect_available = True
except Exception:
    langdetect_available = False

# small ID-word heuristic fallback (keeps it lightweight)
ID_WORDS = {
    "cara","bagaimana","kirim","pesan","jadwal","nomor","akun","whatsapp","fitur",
    "harga","kenapa","tidak","bisa","daftar","hapus","atur","aturan","lanjut"
}

def detect_language(text):
    """
    Return "id" for Indonesian, "en" for English (fallback).
    Uses langdetect if available, otherwise a heuristic.
    """
    text = str(text).lower().strip()
    if not text:
        return "en"
    if langdetect_available:
        try:
            lang = detect(text)
            if lang.startswith("id") or lang == "ms":
                return "id"
            else:
                return "en"
        except Exception:
            pass
    # fallback: count ID words
    tokens = text.split()
    if not tokens:
        return "en"
    id_count = sum(1 for t in tokens if t in ID_WORDS)
    # if >=30% tokens look Indonesian -> id
    if id_count >= max(1, int(0.3 * len(tokens))):
        return "id"
    return "en"

# ---------------------------
# Similarity helpers
# ---------------------------
def cosine_sim_tfidf(query):
    q_vec = tfidf_vectorizer.transform([preprocess(query)])
    return linear_kernel(q_vec, tfidf_matrix).flatten()

def cosine_sim_embeddings(query_emb, all_embs):
    qn = query_emb / (np.linalg.norm(query_emb)+1e-10)
    an = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True)+1e-10)
    return an.dot(qn)

# ---------------------------
# Map to final bilingual response
# ---------------------------
def get_response_for_intent(intent_key, user_lang):
    """
    Return the response string for given intent and language.
    Fallback order: requested language -> id -> en -> generic message.
    """
    entry = responses_bilingual.get(intent_key, {})
    # some files might use 'id'/'en' or 'ID'/'EN' — normalize keys:
    entry_norm = {k.lower(): v for k,v in entry.items()}
    resp = entry_norm.get(user_lang)
    if resp:
        return resp
    # fallback
    return entry_norm.get("id") or entry_norm.get("en") or "Maaf, saya belum punya jawaban untuk itu."

# ---------------------------
# Hybrid predict function
# ---------------------------
def predict_intent(query, top_k=3, weight_embed=0.65, weight_tfidf=0.35):
    """
    Returns list of candidates: {intent, score, tfidf_score, embed_score, example, response_lang_map}
    """
    q_p = preprocess(query)
    tfidf_scores = cosine_sim_tfidf(q_p)
    if embedding_available:
        q_emb = model.encode([q_p], convert_to_numpy=True)[0]
        embed_scores = cosine_sim_embeddings(q_emb, embeddings)
    else:
        embed_scores = np.zeros_like(tfidf_scores)

    # normalize to [0,1]
    def norm(a):
        a = np.array(a)
        if a.max() - a.min() < 1e-12:
            return np.zeros_like(a)
        return (a - a.min()) / (a.max() - a.min())

    tfidf_n = norm(tfidf_scores)
    embed_n = norm(embed_scores)
    combined = weight_embed * embed_n + weight_tfidf * tfidf_n

    top_idx = np.argsort(combined)[::-1][:top_k]
    results = []
    for idx in top_idx:
        intent_key = intents[idx]
        results.append({
            "intent": intent_key,
            "score": float(combined[idx]),
            "tfidf_score": float(tfidf_scores[idx]),
            "embed_score": float(embed_scores[idx]) if embedding_available else None,
            "example": texts[idx]
        })
    return results

# ---------------------------
# Interactive Chat CLI
# ---------------------------
def chat_cli():
    print("\n=== Chattea Hybrid RAG Chatbot (bilingual) ===")
    print("Type 'quit' to exit. Bot will reply in the same language you use (ID/EN).")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        candidates = predict_intent(q, top_k=3)
        if not candidates:
            print("Maaf, saya tidak mengerti. Bisa diperjelas?")
            continue

        # choose top candidate
        best = candidates[0]
        user_lang = detect_language(q)
        reply = get_response_for_intent(best["intent"], user_lang)
        print(f"\n[intent: {best['intent']}, score: {best['score']:.3f}]")
        print(reply)

# ---------------------------
# Simple evaluation utility (optional)
# ---------------------------
def evaluate_simple(test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(texts, intents, test_size=test_size, random_state=42)
    # TF-IDF baseline (train on X_train)
    v = TfidfVectorizer(ngram_range=(1,2))
    X_train_mat = v.fit_transform(X_train)
    def predict_tfidf_only(q):
        sims = linear_kernel(v.transform([preprocess(q)]), X_train_mat).flatten()
        return y_train[sims.argmax()]

    y_pred = [predict_tfidf_only(q) for q in X_test]
    acc_tfidf = metrics.accuracy_score(y_test, y_pred)
    print(f"\nTF-IDF baseline accuracy: {acc_tfidf:.4f} on {len(y_test)} samples")

    # Hybrid evaluation (if embeddings available)
    if embedding_available:
        emb_all = model.encode(X_train + X_test, convert_to_numpy=True)
        emb_train = emb_all[:len(X_train)]
        def predict_hybrid(q):
            q_p = preprocess(q)
            q_emb = model.encode([q_p])[0]
            sim_emb = cosine_sim_embeddings(q_emb, emb_train)
            sim_tfidf = linear_kernel(v.transform([q_p]), X_train_mat).flatten()

            def nrm(a):
                a = np.array(a)
                if a.max() - a.min() < 1e-12:
                    return np.zeros_like(a)
                return (a - a.min()) / (a.max() - a.min())

            comb = 0.65 * nrm(sim_emb) + 0.35 * nrm(sim_tfidf)
            return y_train[comb.argmax()]

        y_pred_h = [predict_hybrid(q) for q in X_test]
        acc_h = metrics.accuracy_score(y_test, y_pred_h)
        print(f"Hybrid TF-IDF+Embedding accuracy: {acc_h:.4f}")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("Starting Chattea hybrid RAG (unified dataset + bilingual responses).")
    try:
        evaluate_simple()
    except Exception:
        print("Evaluation skipped/failed (ok).")
    chat_cli()
