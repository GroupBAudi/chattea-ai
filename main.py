#!/usr/bin/env python3
"""
Chattea Intent Classifier - CNN + Word2Vec (Optimized)
Run with: python main.py

Required files:
- chatbot_dataset.csv (text, intent columns)
- responses.json

First run: Trains Word2Vec and CNN, saves to:
  - word2vec.model (word embeddings)
  - chattea.pth (CNN classifier)
Subsequent runs: Loads pre-trained models for instant inference
"""

import json
import pandas as pd
import re
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # File paths
    DATASET_PATH = "chatbot_dataset.csv"
    RESPONSES_PATH = "responses.json"
    MODEL_PATH = "chattea.pth"
    WORD2VEC_PATH = "word2vec.model"
    
    # Word2Vec parameters (OPTIMIZED)
    EMBEDDING_DIM = 100        # Embedding dimension
    WORD2VEC_WINDOW = 5        # Context window
    WORD2VEC_MIN_COUNT = 1     # Minimum word frequency
    WORD2VEC_SG = 1            # Skip-gram (better for small datasets)
    
    # CNN parameters (OPTIMIZED)
    NUM_FILTERS = 128          # Filters per kernel
    KERNEL_SIZES = [2, 3, 4]   # Includes 2-word phrases!
    DROPOUT = 0.5              # Higher regularization
    MAX_SEQ_LENGTH = 20        # Shorter = more efficient
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Inference parameters
    FUZZY_CUTOFF = 0.8
    CONFIDENCE_THRESHOLD = 0.75
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)

# ============================================================================
# DEVICE SETUP
# ============================================================================

print("=" * 80)
print("CHATTEA INTENT CLASSIFIER - CNN + WORD2VEC")
print("=" * 80)
print(f"Device: {config.DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# ============================================================================
# TEXT PROCESSING
# ============================================================================

def clean_text(text):
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

def tokenize(text):
    """Tokenize text into words"""
    return clean_text(text).split()

def build_vocabulary(texts):
    """Extract all unique words from texts"""
    vocab = set()
    for text in texts:
        vocab.update(re.findall(r'\w+', str(text).lower()))
    return vocab

def fuzzy_correct(text, vocab, cutoff=Config.FUZZY_CUTOFF):
    """Correct typos using Levenshtein distance"""
    words = re.findall(r'\w+', text.lower())
    corrected = []
    
    for word in words:
        matches = get_close_matches(word, vocab, n=1, cutoff=cutoff)
        corrected.append(matches[0] if matches else word)
    
    return ' '.join(corrected)

# ============================================================================
# WORD2VEC EMBEDDER
# ============================================================================

class Word2VecEmbedder:
    """Word2Vec embedding wrapper with proper initialization"""
    
    def __init__(self):
        self.model = None
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {}
        self.embedding_matrix = None
        self.vocab_size = 0
        self.embed_dim = Config.EMBEDDING_DIM
    
    def train(self, sentences):
        """Train Word2Vec on tokenized sentences"""
        print("\n🧠 Training Word2Vec...")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=Config.EMBEDDING_DIM,
            window=Config.WORD2VEC_WINDOW,
            min_count=Config.WORD2VEC_MIN_COUNT,
            sg=Config.WORD2VEC_SG,  # Skip-gram (better for small data!)
            seed=Config.RANDOM_SEED,
            workers=4
        )
        
        # Build word-to-index mapping
        idx = 2
        for word in self.model.wv.index_to_key:
            self.word2idx[word] = idx
            idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Create embedding matrix (CRITICAL FOR PERFORMANCE!)
        self.embedding_matrix = np.zeros(
            (self.vocab_size, self.embed_dim), 
            dtype=np.float32
        )
        
        for word, idx in self.word2idx.items():
            if word in ['<PAD>', '<UNK>']:
                # Keep as zeros for padding, small random for unknown
                if word == '<UNK>':
                    self.embedding_matrix[idx] = np.random.randn(self.embed_dim) * 0.01
            else:
                try:
                    self.embedding_matrix[idx] = self.model.wv[word]
                except KeyError:
                    self.embedding_matrix[idx] = np.random.randn(self.embed_dim) * 0.01
        
        print(f"✓ Word2Vec trained: vocab={self.vocab_size}, dim={self.embed_dim}")
        return self
    
    def save(self, path=Config.WORD2VEC_PATH):
        """Save Word2Vec model"""
        if self.model:
            self.model.save(path)
            print(f"✓ Word2Vec saved to {path}")
    
    def load(self, path=Config.WORD2VEC_PATH):
        """Load Word2Vec model"""
        print(f"\n🧠 Loading Word2Vec from {path}...")
        self.model = Word2Vec.load(path)
        
        # Rebuild mappings and matrix
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        idx = 2
        for word in self.model.wv.index_to_key:
            self.word2idx[word] = idx
            idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        self.embedding_matrix = np.zeros(
            (self.vocab_size, self.embed_dim),
            dtype=np.float32
        )
        
        for word, idx in self.word2idx.items():
            if word in ['<PAD>', '<UNK>']:
                if word == '<UNK>':
                    self.embedding_matrix[idx] = np.random.randn(self.embed_dim) * 0.01
            else:
                try:
                    self.embedding_matrix[idx] = self.model.wv[word]
                except KeyError:
                    self.embedding_matrix[idx] = np.random.randn(self.embed_dim) * 0.01
        
        print(f"✓ Word2Vec loaded: vocab={self.vocab_size}, dim={self.embed_dim}")
        return self
    
    def encode_sequence(self, tokens, max_length=Config.MAX_SEQ_LENGTH):
        """Convert tokens to indices"""
        indices = [
            self.word2idx.get(token, self.word2idx["<UNK>"]) 
            for token in tokens[:max_length]
        ]
        
        # Pad to max_length
        while len(indices) < max_length:
            indices.append(self.word2idx["<PAD>"])
        
        return indices
    
    def sentence_vector(self, tokens):
        """Get average sentence vector for retrieval"""
        vectors = []
        for token in tokens:
            if token in self.word2idx and token not in ("<PAD>", "<UNK>"):
                idx = self.word2idx[token]
                if idx < len(self.embedding_matrix):
                    vectors.append(self.embedding_matrix[idx])
        
        if len(vectors) == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        return np.mean(vectors, axis=0)

# ============================================================================
# CNN MODEL ARCHITECTURE
# ============================================================================

class TextCNN(nn.Module):
    """
    CNN for Text Classification (Kim, 2014)
    
    KEY IMPROVEMENT: Initialize embeddings with Word2Vec weights!
    """
    
    def __init__(self, vocab_size, embedding_dim, num_classes, 
                 embedding_matrix=None):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CRITICAL: Initialize with Word2Vec weights!
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(
                torch.tensor(embedding_matrix, dtype=torch.float32)
            )
            print("✓ CNN initialized with Word2Vec embeddings!")
        
        # Multiple convolution layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=Config.NUM_FILTERS,
                kernel_size=k
            )
            for k in Config.KERNEL_SIZES
        ])
        
        # Dropout
        self.dropout = nn.Dropout(Config.DROPOUT)
        
        # Output layer
        self.fc = nn.Linear(
            Config.NUM_FILTERS * len(Config.KERNEL_SIZES),
            num_classes
        )
    
    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        # Embed: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        
        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            # Conv: (batch, embed_dim, seq_len) -> (batch, num_filters, L)
            conv_out = F.relu(conv(embedded))
            # Max pool: (batch, num_filters, L) -> (batch, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate: (batch, num_filters * 3)
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout + classification
        dropped = self.dropout(concatenated)
        logits = self.fc(dropped)
        
        return logits

# ============================================================================
# DATASET
# ============================================================================

class IntentDataset(Dataset):
    """Simple dataset wrapper"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """Train the CNN classifier"""
    print("\n" + "=" * 80)
    print("TRAINING CNN MODEL")
    print("=" * 80)
    
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders
    train_dataset = IntentDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True
    )
    
    best_val_acc = 0
    
    print("\nEpoch | Train Acc | Train Loss | Val Acc | Val Loss")
    print("-" * 65)
    
    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(config.DEVICE)
            batch_y = batch_y.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(config.DEVICE)
            y_val_device = y_val.to(config.DEVICE)
            
            val_outputs = model(X_val_device)
            val_loss = criterion(val_outputs, y_val_device).item()
            val_acc = (val_outputs.argmax(1) == y_val_device).float().mean().item()
        
        # Print progress
        if epoch % 5 == 0 or epoch == Config.EPOCHS - 1:
            print(f"{epoch:5d} | {train_acc:9.4f} | {avg_train_loss:10.4f} | {val_acc:7.4f} | {val_loss:8.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
    
    print("\n" + "=" * 80)
    print(f"✓ Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"✓ Model saved to: {Config.MODEL_PATH}")
    print("=" * 80)
    
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    
    return model

# ============================================================================
# CHATBOT CLASS
# ============================================================================

class ChatteaBot:
    """Main chatbot class with hybrid classification"""
    
    def __init__(self, model, embedder, label_encoder, responses, 
                 df, sentence_vectors, vocab):
        self.model = model
        self.embedder = embedder
        self.le = label_encoder
        self.responses = responses
        self.df = df.reset_index(drop=True)
        self.sentence_vectors = sentence_vectors.astype(np.float32)
        self.vocab = vocab
        
        # Intent mapping
        self.intent_map = {i: label for i, label in enumerate(self.le.classes_)}
        
        self.model.eval()
    
    def _get_response(self, intent):
        """Get response for intent"""
        response = self.responses.get(
            intent, 
            self.responses.get("help", "I'm not sure how to help with that.")
        )
        
        if isinstance(response, dict):
            return response.get("en", response.get("id", next(iter(response.values()))))
        
        return response
    
    def get_reply(self, user_input):
        """Get chatbot response"""
        text = str(user_input).strip()
        
        if text == "":
            return "Say something :)"
        
        # Rule-based greeting
        if any(g in text.lower() for g in ["hai", "halo", "hello", "hi", "hey", "pagi", "siang", "malam"]):
            return self._get_response("greeting")
        
        # Fuzzy correction
        corrected = fuzzy_correct(text, self.vocab, Config.FUZZY_CUTOFF)
        tokens = tokenize(corrected)
        
        # Model prediction
        sequence = self.embedder.encode_sequence(tokens, Config.MAX_SEQ_LENGTH)
        x = torch.tensor([sequence], dtype=torch.long)        # create on CPU first
        x = x.to(config.DEVICE)                                # move explicitly
        
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            model_conf = float(probs.max())
            model_idx = int(np.argmax(probs))
            model_intent = self.intent_map[model_idx]
        
        # Retrieval fallback
        user_vec = self.embedder.sentence_vector(tokens).reshape(1, -1)
        
        if self.sentence_vectors is None or len(self.sentence_vectors) == 0:
            retrieval_intent = model_intent
            retrieval_score = 0.0
        else:
            similarities = cosine_similarity(user_vec, self.sentence_vectors)[0]
            best_idx = int(np.argmax(similarities))
            retrieval_score = float(similarities[best_idx])
            retrieval_intent = str(self.df.iloc[best_idx]["intent"])
        
        # Decision
        if model_conf >= Config.CONFIDENCE_THRESHOLD:
            final_intent = model_intent
            decision = "MODEL"
        else:
            final_intent = retrieval_intent
            decision = "RETRIEVAL"
        
        return self._get_response(final_intent)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Load data
    print("\n📂 Loading data...")
    if not os.path.exists(Config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {Config.DATASET_PATH}")
    
    df = pd.read_csv(Config.DATASET_PATH)
    
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("Dataset must have 'text' and 'intent' columns")
    
    print(f"✓ Loaded {len(df)} samples, {df['intent'].nunique()} intents")
    
    # Load responses
    if not os.path.exists(Config.RESPONSES_PATH):
        raise FileNotFoundError(f"Responses file not found: {Config.RESPONSES_PATH}")
    
    with open(Config.RESPONSES_PATH, "r", encoding="utf-8") as f:
        responses = json.load(f)
    
    # Build vocabulary
    print("\n📚 Building vocabulary...")
    vocab = build_vocabulary(df['text'].tolist())
    print(f"✓ Vocabulary: {len(vocab)} words")
    
    # Label encoding
    print("\n🏷️  Encoding labels...")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intent'].astype(str))
    num_classes = len(le.classes_)
    print(f"✓ Classes: {num_classes}")
    
    # Tokenize
    print("\n✂️  Tokenizing...")
    df['tokens'] = df['text'].apply(lambda t: tokenize(str(t)))
    
    # Word2Vec
    embedder = Word2VecEmbedder()
    if os.path.exists(Config.WORD2VEC_PATH):
        embedder.load(Config.WORD2VEC_PATH)
    else:
        embedder.train(df['tokens'].tolist())
        embedder.save(Config.WORD2VEC_PATH)
    
    # Prepare sequences
    print("\n📊 Preparing sequences...")
    sequences = np.array([
        embedder.encode_sequence(tokens, Config.MAX_SEQ_LENGTH)
        for tokens in df['tokens']
    ], dtype=np.int64)
    
    X = torch.tensor(sequences, dtype=torch.long)  # Keep on CPU
    y = torch.tensor(df['label'].values, dtype=torch.long)
    
    # Train/val split
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=df['label']
    )
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    
    # Build model
    model = TextCNN(
        vocab_size=embedder.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        num_classes=num_classes,
        embedding_matrix=embedder.embedding_matrix
    )
    
    # Train or load model
    if os.path.exists(Config.MODEL_PATH):
        print(f"\n✓ Found existing model: {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=config.DEVICE))
        model.eval()
        print("✓ Model loaded!")
    else:
        print("\n⚠️  No pre-trained model found. Training from scratch...")
        model = train_model(model, X_train, y_train, X_val, y_val)
    
    # Prepare sentence vectors for retrieval
    print("\n📐 Preparing sentence vectors for retrieval...")

    sent_vecs = np.array([
        embedder.sentence_vector(tokens)
        for tokens in df["tokens"]
    ], dtype=np.float32)


    # Create chatbot instance
    print("\n🤖 Initializing ChatteaBot...")
    bot = ChatteaBot(
        model=model,
        embedder=embedder,
        label_encoder=le,
        responses=responses,
        df=df,
        sentence_vectors=sent_vecs,
        vocab=vocab
    )

    print("\n✓ ChatteaBot ready!")
    print("Type something, or type 'exit' to stop.\n")

    # Interactive loop
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["q", "exit", "quit", "keluar", "stop"]:
            print("Chattea: Goodbye!")
            break

        reply = bot.get_reply(user_input)

        print("Chattea:", reply)


# Run main()
if __name__ == "__main__":
    main()
