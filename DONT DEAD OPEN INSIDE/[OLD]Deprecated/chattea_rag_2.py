"""
chattea_rag_hybrid.py
TF-IDF + Sentence-Transformer Hybrid RAG pipeline for Chattea dataset.

Usage:
    python chattea_rag_hybrid.py           # runs build + evaluation + interactive demo
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

# Local dataset paths (you uploaded these)
CSV_PATH = "/mnt/data/chattea.csv"
RESPONSES_PATH = "/mnt/data/responses.json"

# Output artifact paths
OUT_DIR = "/mnt/data"
TFIDF_VECTORIZER_PATH = os.path.join(OUT_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_PATH = os.path.join(OUT_DIR, "tfidf_matrix.npz")  # optional (scipy sparse)
TEXTS_INTENTS_PATH = os.path.join(OUT_DIR, "texts_intents.json")
EMBEDDINGS_PATH = os.path.join(OUT_DIR, "embeddings.npy")
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # change if desired

# Preprocessing
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)         # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()    # collapse spaces
    return text

# ---------------------------
# 1) Load dataset & responses
# ---------------------------
import pandas as pd
df = pd.read_csv(CSV_PATH)
with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
    responses = json.load(f)

# quick check
print("Loaded CSV rows:", len(df))
print("Sample columns:", df.columns.tolist())

texts = df["text"].astype(str).apply(preprocess).tolist()
intents = df["intent"].astype(str).tolist()

# Build a mapping list for index -> (text, intent)
index_to_item = [{"text": t, "intent": it} for t, it in zip(texts, intents)]

# Save the mapping (useful for later)
with open(TEXTS_INTENTS_PATH, "w", encoding="utf-8") as f:
    json.dump(index_to_item, f, indent=2, ensure_ascii=False)
print(f"Saved texts/intents mapping to: {TEXTS_INTENTS_PATH}")

# ---------------------------
# 2) Build TF-IDF
# ---------------------------
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# persist tfidf vectorizer
with open(TFIDF_VECTORIZER_PATH, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"Saved TF-IDF vectorizer to: {TFIDF_VECTORIZER_PATH}")

# you can optionally save tfidf_matrix as sparse .npz if desired, but not required
try:
    from scipy import sparse
    sparse.save_npz(os.path.join(OUT_DIR, "tfidf_matrix.npz"), tfidf_matrix)
    print("Saved TF-IDF matrix to tfidf_matrix.npz")
except Exception as e:
    print("scipy not available or save failed:", e)

# ---------------------------
# 3) Try to build embeddings
# ---------------------------
embeddings = None
embedding_dim = None
embedding_available = False

try:
    from sentence_transformers import SentenceTransformer, util
    print("Attempting to load SentenceTransformer model:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # encode all texts (batch)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embedding_dim = embeddings.shape[1]
    embedding_available = True
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to: {EMBEDDINGS_PATH} (shape: {embeddings.shape})")
except Exception as e:
    print("SentenceTransformers embedding failed or not available in this environment.")
    print("Error:", e)
    print("Continuing with TF-IDF only. To enable embeddings, run this script where sentence-transformers is installed and model can be downloaded.")
    embedding_available = False

# ---------------------------
# Utility: cosine similarity helpers
# ---------------------------
def cosine_sim_embeddings(query_emb, all_embs):
    # Normalize and compute dot product
    # query_emb: (D,), all_embs: (N, D)
    # returns array of shape (N,)
    qn = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    an = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-10)
    sims = an.dot(qn)
    return sims

def cosine_sim_tfidf(query, tfidf_matrix=tfidf_matrix, vectorizer=tfidf_vectorizer):
    q_vec = vectorizer.transform([preprocess(query)])
    # linear_kernel gives cosine similarity for tf-idf
    sims = linear_kernel(q_vec, tfidf_matrix).flatten()
    return sims

# ---------------------------
# Predict function (hybrid)
# ---------------------------
from random import choice

def predict_intent(query, top_k=3, weight_embed=0.65, weight_tfidf=0.35):
    """
    Returns list of candidates: [(intent, combined_score, response, text_example), ...]
    If embeddings aren't available, uses TF-IDF only.
    """
    query_pre = preprocess(query)
    tfidf_scores = cosine_sim_tfidf(query_pre)
    
    embed_scores = None
    if embedding_available:
        q_emb = model.encode([query_pre], convert_to_numpy=True)[0]
        embed_scores = cosine_sim_embeddings(q_emb, embeddings)
    else:
        # If no embeddings, set embed_scores to zero array
        embed_scores = np.zeros_like(tfidf_scores)

    # normalize scores to 0-1 for combining
    def norm(arr):
        arr = np.array(arr)
        if arr.max() - arr.min() < 1e-12:
            return np.zeros_like(arr)
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        return arr

    tfidf_n = norm(tfidf_scores)
    embed_n = norm(embed_scores) if embedding_available else np.zeros_like(tfidf_n)

    combined = weight_embed * embed_n + weight_tfidf * tfidf_n

    top_idx = np.argsort(combined)[::-1][:top_k]
    results = []
    for idx in top_idx:
        intent = intents[idx]
        example_text = texts[idx]
        resp = responses.get(intent, "Sorry, I don't have an answer for that yet.")
        results.append({
            "intent": intent,
            "score": float(combined[idx]),
            "tfidf_score": float(tfidf_scores[idx]),
            "embed_score": float(embed_scores[idx]) if embedding_available else None,
            "example": example_text,
            "response": resp
        })
    return results

# Quick smoke test
print("\nQuick test predictions (examples):")
for q in ["how to create account", "kirim pesan massal", "how to schedule messages", "how to check whatsapp number"]:
    print("\nQuery:", q)
    preds = predict_intent(q, top_k=2)
    pprint(preds[:2])

# ---------------------------
# 4) Evaluation (optional)
# ---------------------------
# If you want, we can compute a simple accuracy on held-out split:
def evaluate_simple(test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(texts, intents, test_size=test_size, random_state=random_state)
    # Build vectorizer on train set only
    v = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X_train_tfidf = v.fit_transform(X_train)
    X_test_tfidf = v.transform(X_test)

    # Embedding train/test if available
    emb_train = emb_test = None
    if embedding_available:
        emb_all = model.encode(X_train + X_test, convert_to_numpy=True)
        emb_train = emb_all[:len(X_train)]
        emb_test = emb_all[len(X_train):]

    # Predict using TF-IDF only baseline
    def predict_tfidf_only(q):
        qv = v.transform([preprocess(q)])
        sims = linear_kernel(qv, X_train_tfidf).flatten()
        idx = sims.argmax()
        return y_train[idx]

    y_pred_tfidf = [predict_tfidf_only(q) for q in X_test]
    acc_tfidf = metrics.accuracy_score(y_test, y_pred_tfidf)

    print(f"\nTF-IDF baseline Accuracy: {acc_tfidf:.4f} ({len(y_test)} samples)")

    if embedding_available:
        # hybrid prediction on test
        def predict_hybrid(q, w_emb=0.6, w_tfidf=0.4):
            # compute similarity against training set embeddings and tfidf
            q_pre = preprocess(q)
            q_emb = model.encode([q_pre], convert_to_numpy=True)[0]
            sim_emb = cosine_sim_embeddings(q_emb, emb_train)
            qv = v.transform([q_pre])
            sim_tfidf = linear_kernel(qv, X_train_tfidf).flatten()

            # normalize both
            def normalize(a):
                a = np.array(a)
                if a.max()-a.min() < 1e-12:
                    return np.zeros_like(a)
                return (a - a.min()) / (a.max() - a.min())
            ne = normalize(sim_emb)
            nt = normalize(sim_tfidf)
            comb = w_emb * ne + w_tfidf * nt
            idx = comb.argmax()
            return y_train[idx]

        y_pred_hybrid = [predict_hybrid(q) for q in X_test]
        acc_hybrid = metrics.accuracy_score(y_test, y_pred_hybrid)
        print(f"Hybrid (embedding+tfidf) Accuracy: {acc_hybrid:.4f}")

# run evaluation
try:
    evaluate_simple()
except Exception as e:
    print("Evaluation failed:", e)

# ---------------------------
# 5) Interactive CLI
# ---------------------------
def interactive_cli():
    print("\n=== Chattea Hybrid RAG Chatbot ===")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        preds = predict_intent(q, top_k=3)
        best = preds[0]
        print(f"\nDetected intent: {best['intent']} (score {best['score']:.3f})")
        print("Response:\n", best["response"])

if __name__ == "__main__":
    # Launch CLI if run directly
    interactive_cli()
