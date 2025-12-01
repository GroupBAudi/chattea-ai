#!/usr/bin/env python3
"""
Chattea Intent Classifier - CNN + Word2Vec (Interactive)
- Debug table, retrieval fallback
- Retrieval fallback uses average Word2Vec sentence vectors (cosine similarity)
- Fuzzy correction, rule-based greetings, model & Word2Vec save/load

Files expected:
 - chatbot_dataset.csv   (columns: text,intent)
 - responses.json

Usage:
    python chattea_cnn_word2vec.py
"""

import os
import re
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from difflib import get_close_matches
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG
# -------------------------
class Config:
    DATASET_PATH = "chatbot_dataset.csv"
    RESPONSES_PATH = "responses.json"
    MODEL_PATH = "cnn_word2vec.pth"
    WORD2VEC_PATH = "word2vec.model"

    EMBEDDING_DIM = 100           # Word2Vec dim
    MAX_SEQUENCE_LENGTH = 20      # words
    KERNEL_SIZES = [2, 3, 4]
    NUM_FILTERS = 128
    DROPOUT = 0.5

    BATCH_SIZE = 32
    EPOCHS = 30
    LR = 1e-3
    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    FUZZY_CUTOFF = 0.8
    CONFIDENCE_THRESHOLD = 0.90

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)


# -------------------------
# UTIL: Preprocessing & Fuzzy
# -------------------------
def clean_text(text: str):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return clean_text(text).split()

def build_vocab_from_texts(texts):
    vocab = set()
    for t in texts:
        vocab.update(re.findall(r'\w+', str(t).lower()))
    return vocab

def fuzzy_correct(text, vocab, cutoff=Config.FUZZY_CUTOFF):
    words = re.findall(r'\w+', text.lower())
    corrected_words = []
    for w in words:
        matches = get_close_matches(w, list(vocab), n=1, cutoff=cutoff)
        corrected_words.append(matches[0] if matches else w)
    # reconstruct but keep original spacing/punctuation removed earlier
    return " ".join(corrected_words)


# -------------------------
# Word2Vec embedder wrapper
# -------------------------
class WordEmbedder:
    def __init__(self, embedding_dim=Config.EMBEDDING_DIM, min_count=1, window=5):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.window = window
        self.model = None
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {}
        self.embedding_matrix = None

    def train(self, sentences):
        print("Training Word2Vec...")
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            sg=1,
            seed=Config.RANDOM_SEED,
            workers=4
        )
        # build mappings
        idx = 2
        for w in self.model.wv.index_to_key:
            self.word2idx[w] = idx
            idx += 1
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}
        vocab_size = len(self.word2idx)
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        for w, i in self.word2idx.items():
            if w in ['<PAD>', '<UNK>']:
                continue
            try:
                self.embedding_matrix[i] = self.model.wv[w]
            except KeyError:
                self.embedding_matrix[i] = np.random.randn(self.embedding_dim) * 0.01
        print(f"✓ Word2Vec trained: vocab={vocab_size}, dim={self.embedding_dim}")
        return self

    def save(self, path=Config.WORD2VEC_PATH):
        if self.model:
            self.model.save(path)

    def load(self, path=Config.WORD2VEC_PATH):
        self.model = Word2Vec.load(path)
        # rebuild mappings & matrix
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for w in self.model.wv.index_to_key:
            self.word2idx[w] = idx
            idx += 1
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}
        vocab_size = len(self.word2idx)
        self.embedding_matrix = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        for w, i in self.word2idx.items():
            if w in ['<PAD>', '<UNK>']:
                continue
            try:
                self.embedding_matrix[i] = self.model.wv[w]
            except KeyError:
                self.embedding_matrix[i] = np.random.randn(self.embedding_dim) * 0.01
        print(f"✓ Word2Vec loaded: {path}")
        return self

    def encode_sequence(self, tokens, max_length=Config.MAX_SEQUENCE_LENGTH):
        indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens[:max_length]]
        # pad
        while len(indices) < max_length:
            indices.append(self.word2idx["<PAD>"])
        return indices

    def sentence_vector(self, tokens):
        # average word vectors; handle unknowns
        vecs = []
        for t in tokens:
            if t in self.word2idx and t not in ("<PAD>",):
                idx = self.word2idx.get(t, self.word2idx["<UNK>"])
                # if embedding_matrix length, safe index
                if idx < len(self.embedding_matrix):
                    vecs.append(self.embedding_matrix[idx])
        if len(vecs) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        return np.mean(vecs, axis=0)


# -------------------------
# CNN model (TextCNN)
# -------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 embedding_matrix=None, kernel_sizes=Config.KERNEL_SIZES,
                 num_filters=Config.NUM_FILTERS, dropout=Config.DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                       # (batch, seq_len, dim)
        emb = emb.permute(0, 2, 1)                    # (batch, dim, seq_len) for Conv1d
        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))                     # (batch, num_filters, L_out)
            pooled = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(2)  # (batch, num_filters)
            conv_outs.append(pooled)
        cat = torch.cat(conv_outs, dim=1)             # (batch, num_filters * len(kernels))
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits


# -------------------------
# Dataset wrapper
# -------------------------
class IntentDataset(Dataset):
    def __init__(self, X, y):
        # X: LongTensor (N, seq_len), y: LongTensor (N,)
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -------------------------
# TRAINING UTIL
# -------------------------
def train_cnn(model, X_train, y_train, X_val, y_val):
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0

    train_ds = IntentDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total if total > 0 else 0.0
        # val
        model.eval()
        with torch.no_grad():
            xb = X_val.to(Config.DEVICE)
            yb = y_val.to(Config.DEVICE)
            logits = model(xb)
            val_loss = float(criterion(logits, yb).item())
            val_acc = float((logits.argmax(1) == yb).float().mean().item())
        if epoch % 5 == 0 or epoch == Config.EPOCHS:
            print(f"Epoch {epoch:3d}/{Config.EPOCHS} | Train Acc: {train_acc:.3f} | Train Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f} | Val Loss: {val_loss:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Training complete. Best Val Acc: {best_val:.4f}")
    # load best
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
    model.eval()
    return model


# -------------------------
# Debug printing (same format)
# -------------------------
def print_debug(query, model_intent, model_conf, retrieval_intent, retrieval_score, final_intent, decision):
    print("\n" + "=" * 80)
    print(f"QUERY         : {query}")
    print(f"Model Predict : {model_intent:<20} Confidence: {model_conf:.4f} ({model_conf*100:6.2f}%)")
    print(f"Threshold     : {Config.CONFIDENCE_THRESHOLD} → Use Model?: {'YES' if model_conf > Config.CONFIDENCE_THRESHOLD else 'NO'}")
    print(f"Retrieval     : {retrieval_intent:<20} Score: {retrieval_score:.4f}")
    print(f"FINAL INTENT  : → {final_intent} ← (Source: {decision})")
    print("=" * 80)


# -------------------------
# Main interactive bot class
# -------------------------
class ChatteaBot:
    def __init__(self, model, embedder: WordEmbedder, label_encoder: LabelEncoder,
                 responses: dict, df: pd.DataFrame, sent_vectors: np.ndarray, vocab:set):
        self.model = model
        self.embedder = embedder
        self.le = label_encoder
        self.responses = responses
        self.df = df.reset_index(drop=True)
        self.sent_vectors = sent_vectors.astype(np.float32)
        self.vocab = vocab
        # mapping index -> intent name
        self.intent_map = {i: label for i, label in enumerate(self.le.classes_)}

    def _get_response(self, intent):
        r = self.responses.get(intent, self.responses.get("help", "I'm not sure how to help with that."))
        if isinstance(r, dict):
            return r.get("en", next(iter(r.values())))
        return r

    def get_reply(self, user_input, debug=False):
        text = str(user_input).strip()
        if text == "":
            return "Say something :)"

        # rule-based greeting (same words as in MiniLM script)
        if any(g in text.lower() for g in ["hai", "halo", "hello", "hi", "hey", "pagi", "siang", "malam"]):
            if debug:
                print_debug(user_input, "greeting", 1.0, "greeting", 1.0, "greeting", "RULE-BASED")
            return self._get_response("greeting")

        # fuzzy correction
        corrected = fuzzy_correct(text, self.vocab, cutoff=Config.FUZZY_CUTOFF)
        tokens = tokenize(corrected)

        # model prediction
        seq = self.embedder.encode_sequence(tokens, Config.MAX_SEQUENCE_LENGTH)
        x = torch.LongTensor([seq]).to(Config.DEVICE)
        with torch.no_grad():
            logits = self.model(x)                         # shape (1, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            model_conf = float(probs.max())
            model_idx = int(np.argmax(probs))
            model_intent = self.intent_map[model_idx]

        # retrieval fallback - cosine with average word2vec sentence vectors
        user_vec = self.embedder.sentence_vector(tokens).reshape(1, -1)
        # compute cosine similarity to stored sentence vectors
        # if no vectors, fallback to zero
        if self.sent_vectors is None or len(self.sent_vectors) == 0:
            retrieval_intent = model_intent
            retrieval_score = 0.0
        else:
            sims = cosine_similarity(user_vec, self.sent_vectors)[0]
            best_idx = int(np.argmax(sims))
            retrieval_score = float(sims[best_idx])
            retrieval_intent = str(self.df.iloc[best_idx]["intent"])

        # decision
        if model_conf >= Config.CONFIDENCE_THRESHOLD:
            final_intent = model_intent
            decision = "MODEL"
        else:
            final_intent = retrieval_intent
            decision = "RETRIEVAL"

        if debug:
            print_debug(user_input, model_intent, model_conf, retrieval_intent, retrieval_score, final_intent, decision)

        return self._get_response(final_intent)


# -------------------------
# MAIN pipeline
# -------------------------
def main():
    print("="*80)
    print("CHATTEA - CNN + Word2Vec Interactive")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print("="*80)

    # load dataset
    if not os.path.exists(Config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {Config.DATASET_PATH}")
    df = pd.read_csv(Config.DATASET_PATH)
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'intent' columns")
    print(f"Loaded dataset: {len(df)} samples, {df['intent'].nunique()} intents")

    # load responses
    if not os.path.exists(Config.RESPONSES_PATH):
        raise FileNotFoundError(f"Responses file not found: {Config.RESPONSES_PATH}")
    with open(Config.RESPONSES_PATH, "r", encoding="utf-8") as f:
        responses = json.load(f)

    # build vocab
    vocab = build_vocab_from_texts(df['text'].tolist())
    print(f"Vocabulary size (approx): {len(vocab)}")

    # encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intent'].astype(str))
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}")

    # preprocess tokens for Word2Vec (clean + tokenized)
    df['tokens'] = df['text'].apply(lambda t: tokenize(str(t)))

    # Word2Vec train/load
    embedder = WordEmbedder(embedding_dim=Config.EMBEDDING_DIM)
    if os.path.exists(Config.WORD2VEC_PATH):
        embedder.load(Config.WORD2VEC_PATH)
    else:
        embedder.train(df['tokens'].tolist())
        embedder.save(Config.WORD2VEC_PATH)

    vocab_size = len(embedder.word2idx)
    print(f"Embedder vocab_size={vocab_size}, emb_dim={Config.EMBEDDING_DIM}")

    # prepare sequences
    sequences = np.array([embedder.encode_sequence(tokens, Config.MAX_SEQUENCE_LENGTH) for tokens in df['tokens']], dtype=np.int64)
    X = torch.LongTensor(sequences)
    y = torch.LongTensor(df['label'].values)

    # train/val split
    train_idx, val_idx = train_test_split(list(range(len(X))), test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED, stratify=y.numpy())
    X_train = X[train_idx].to(Config.DEVICE)
    y_train = y[train_idx].to(Config.DEVICE)
    X_val = X[val_idx].to(Config.DEVICE)
    y_val = y[val_idx].to(Config.DEVICE)

    # build model
    model = TextCNN(vocab_size=vocab_size, embedding_dim=Config.EMBEDDING_DIM, num_classes=num_classes,
                    embedding_matrix=embedder.embedding_matrix).to(Config.DEVICE)

    # train or load model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        model.eval()
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("No pre-trained model found. Training from scratch...")
        model = train_cnn(model, X_train, y_train, X_val, y_val)

    # prepare sentence vectors for retrieval: average word2vec per sentence
    sent_vecs = np.stack([embedder.sentence_vector(tokens) for tokens in df['tokens']])
    # normalize vectors (cosine similarity uses normalized vectors ideally)
    norm = np.linalg.norm(sent_vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    sent_vecs_normed = sent_vecs / norm

    # initialize bot
    bot = ChatteaBot(model, embedder, le, responses, df, sent_vecs_normed, vocab)
    print("✓ Chatbot ready!")

    # basic tests
    print("\n" + "="*80)
    print("TEST QUERIES")
    print("="*80)
    tests = ["hello", "what is chattea", "how to blast message", "check 08123456789", "create instance"]
    for q in tests:
        print(f"\nUser: {q}")
        print("Bot:", bot.get_reply(q, debug=False)[:200])

    # interactive loop (A: match MiniLM format)
    print("\n" + "="*80)
    print("INTERACTIVE MODE (type '/debug <text>' for debug; 'quit' to exit)")
    print("="*80)
    while True:
        try:
            line = input("You: ").strip()
            if line.lower() in ("quit", "exit", "q"):
                print("Goodbye 👋")
                break
            if line == "":
                continue
            if line.startswith("/debug "):
                query = line[len("/debug "):]
                print("\n[DEBUG MODE]")
                reply = bot.get_reply(query, debug=True)
                print("\nBot:", reply)
            else:
                reply = bot.get_reply(line, debug=False)
                print("\nBot:", reply)
        except KeyboardInterrupt:
            print("\nInterrupted — exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    main()
