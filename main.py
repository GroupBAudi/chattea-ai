"""
CHATTEA INTENT CLASSIFICATION SYSTEM
Algorithms: NLP Preprocessing + Word2Vec + CNN + Fuzzy Matching
+ Rule-Based Response System

Author: ChatGPT
Version: Clean-Modular 2025
"""

# ============================
# IMPORTS
# ============================

import os
import re
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec
from difflib import SequenceMatcher


# ============================
# CONFIG
# ============================

class Config:
    DATA_PATH = "chattea_dataset.csv"
    RESPONSES_PATH = "responses.json"

    EMBEDDING_DIM = 100
    MAX_LEN = 20
    CNN_FILTERS = 128
    KERNEL_SIZES = [2, 3, 4]

    BATCH_SIZE = 32
    EPOCHS = 15
    LR = 0.001

    FUZZY_THRESHOLD = 0.75
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    SEED = 42


np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


# ============================
# NLP PREPROCESSING
# ============================

class NLPPreprocessor:
    """Cleans, normalizes, tokenizes text."""

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text):
        return self.clean(text).split()

    def __call__(self, text):
        return self.tokenize(text)


# ============================
# WORD2VEC EMBEDDINGS
# ============================

class WordEmbedder:
    def __init__(self, dim=100):
        self.dim = dim
        self.model = None
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {}
        self.embedding_matrix = None

    def train(self, tokenized_sentences):
        print("Training Word2Vec...")

        self.model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=self.dim,
            min_count=1,
            window=5,
            sg=1
        )

        # Build vocab
        idx = 2
        for word in self.model.wv.index_to_key:
            self.word2idx[word] = idx
            idx += 1

        # Build reverse
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        # Embedding matrix
        vocab_size = len(self.word2idx)
        self.embedding_matrix = np.zeros((vocab_size, self.dim))

        for word, idx in self.word2idx.items():
            if word in ["<PAD>", "<UNK>"]:
                continue
            self.embedding_matrix[idx] = self.model.wv[word]

        print(f"✓ Word2Vec training complete: {vocab_size} tokens")
        return self

    def encode(self, tokens, max_len):
        indices = [self.word2idx.get(t, 1) for t in tokens[:max_len]]
        while len(indices) < max_len:
            indices.append(0)
        return indices


# ============================
# CNN MODEL
# ============================

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, max_len, filters=128, kernels=[2, 3, 4]):
        super().__init__()
        vocab_size, dim = embedding_matrix.shape

        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([
            nn.Conv1d(dim, filters, k)
            for k in kernels
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(filters * len(kernels), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)

        conv_results = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            pooled = torch.max_pool1d(c, c.shape[2]).squeeze(2)
            conv_results.append(pooled)

        out = torch.cat(conv_results, dim=1)
        out = self.dropout(out)
        return self.fc(out)


# ============================
# DATASET WRAPPER
# ============================

class IntentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx])

    def __len__(self):
        return len(self.X)


# ============================
# FUZZY MATCHING
# ============================

class FuzzyMatcher:
    def __init__(self, vocab, threshold=0.75):
        self.vocab = list(vocab)
        self.threshold = threshold

    def correct(self, word):
        best = word
        best_score = 0

        for v in self.vocab:
            score = SequenceMatcher(None, word, v).ratio()
            if score > best_score:
                best_score = score
                best = v

        return best if best_score >= self.threshold else word

    def apply(self, text):
        return " ".join(self.correct(w) for w in text.lower().split())


# ============================
# RULE-BASED RESPONSE SYSTEM
# ============================

class ResponseSystem:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def get(self, intent, lang="en"):
        if intent in self.data:
            return self.data[intent].get(lang, self.data[intent]["en"])
        return "Sorry, I don't understand."


# ============================
# MAIN CLASSIFIER
# ============================

class ChatteaClassifier:
    def __init__(self):
        self.pre = NLPPreprocessor()
        self.embedder = None
        self.model = None
        self.label_map = {}
        self.inv_label_map = {}
        self.fuzzy = None
        self.responses = None

    # ---- data loading ----
    def load_dataset(self, path):
        df = pd.read_csv(path)
        print(f"Loaded dataset: {len(df)} samples")
        return df

    # ---- preprocess ----
    def prepare(self, df):
        df["tokens"] = df["text"].apply(self.pre)

        intents = sorted(df["intent"].unique())
        self.label_map = {i: idx for idx, i in enumerate(intents)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        df["label"] = df["intent"].map(self.label_map)

        train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=Config.SEED)
        val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=Config.SEED)

        return train, val, test

    # ---- embeddings ----
    def build_embeddings(self, train):
        sentences = train["tokens"].tolist()
        self.embedder = WordEmbedder(Config.EMBEDDING_DIM).train(sentences)

        vocab = set(w for s in sentences for w in s)
        self.fuzzy = FuzzyMatcher(vocab, Config.FUZZY_THRESHOLD)

    # ---- encoding ----
    def encode(self, df):
        X = [self.embedder.encode(tokens, Config.MAX_LEN) for tokens in df["tokens"]]
        y = df["label"].values
        return np.array(X), np.array(y)

    # ---- build model ----
    def build_model(self):
        num_classes = len(self.label_map)
        self.model = TextCNN(
            embedding_matrix=self.embedder.embedding_matrix,
            num_classes=num_classes,
            max_len=Config.MAX_LEN,
            filters=Config.CNN_FILTERS,
            kernels=Config.KERNEL_SIZES
        ).to(Config.DEVICE)

    # ---- train ----
    def train(self, train, val):
        X_train, y_train = self.encode(train)
        X_val, y_val = self.encode(val)

        train_loader = DataLoader(IntentDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(IntentDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)

        opt = optim.Adam(self.model.parameters(), lr=Config.LR)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0

        for epoch in range(Config.EPOCHS):
            self.model.train()
            total, correct, total_loss = 0, 0, 0

            for X, y in train_loader:
                X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)

                opt.zero_grad()
                logits = self.model(X)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

                total_loss += loss.item()
                _, pred = torch.max(logits, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            train_acc = correct / total

            # validation
            self.model.eval()
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(Config.DEVICE), y.to(Config.DEVICE)
                    logits = self.model(X)
                    _, pred = torch.max(logits, 1)
                    val_correct += (pred == y).sum().item()
                    val_total += y.size(0)

            val_acc = val_correct / val_total

            print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Acc={train_acc:.3f} | Val Acc={val_acc:.3f}")

            # save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), "best_cnn.pth")

        print("Training complete. Best Val Acc:", best_acc)

    # ---- evaluate ----
    def evaluate(self, test):
        X_test, y_test = self.encode(test)
        loader = DataLoader(IntentDataset(X_test, y_test), batch_size=Config.BATCH_SIZE)

        self.model.load_state_dict(torch.load("best_cnn.pth"))
        self.model.eval()

        preds = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(Config.DEVICE)
                logits = self.model(X)
                _, p = torch.max(logits, 1)
                preds.extend(p.cpu().numpy())

        print("\n=== TEST RESULTS ===")
        print("Accuracy:", accuracy_score(y_test, preds))
        print(classification_report(y_test, preds, target_names=[self.inv_label_map[i] for i in range(len(self.inv_label_map))], zero_division=0))

    # ---- inference ----
    def predict(self, text):
        corrected = self.fuzzy.apply(text)
        tokens = self.pre(corrected)
        encoded = self.embedder.encode(tokens, Config.MAX_LEN)

        x = torch.tensor([encoded], dtype=torch.long).to(Config.DEVICE)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, 1)
            conf, pred = torch.max(probs, 1)

        return self.inv_label_map[pred.item()], conf.item()

    # ---- response system ----
    def load_responses(self):
        self.responses = ResponseSystem(Config.RESPONSES_PATH)

    # ---- CLI demo ----
    def demo(self):
        print("\nChattea Demo — type 'quit' to exit.\n")
        while True:
            text = input("You: ")

            if text.lower() == "quit":
                break

            intent, conf = self.predict(text)
            reply = self.responses.get(intent)

            print(f"\nIntent: {intent} ({conf:.2f})")
            print("Bot:", reply)
            print()


# ============================
# MAIN
# ============================

def main():
    clf = ChatteaClassifier()

    df = clf.load_dataset(Config.DATA_PATH)
    train, val, test = clf.prepare(df)

    clf.build_embeddings(train)
    clf.build_model()
    clf.train(train, val)
    clf.evaluate(test)

    clf.load_responses()
    clf.demo()


if __name__ == "__main__":
    main()
