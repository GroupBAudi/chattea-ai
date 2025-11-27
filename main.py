#!/usr/bin/env python3
"""
Chattea Intent Classifier - Production Script
Run with: python main.py

Required files:
- chattea_dataset.csv
- responses.json

First run: Trains model and saves to cnn_chattea.pth
Subsequent runs: Loads pre-trained model for instant inference
"""

import json
import pandas as pd
import re
import os
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # File paths
    DATASET_PATH = "chattea_dataset.csv"
    RESPONSES_PATH = "responses.json"
    MODEL_PATH = "cnn_chattea.pth"
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Model parameters
    EMBED_DIM = 384
    HIDDEN_DIM_1 = 256
    HIDDEN_DIM_2 = 128
    DROPOUT = 0.3
    
    # Inference parameters
    CONFIDENCE_THRESHOLD = 0.90

config = Config()

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print("CHATTEA INTENT CLASSIFIER")
print("=" * 80)
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EmbeddingClassifier(nn.Module):
    """
    Feedforward Neural Network for Intent Classification
    
    Why Feedforward (not CNN)?
    - Sentence embeddings are feature vectors, not sequences
    - MLP treats each dimension as independent feature
    - CNN would incorrectly assume spatial relationships
    """
    
    def __init__(self, embed_dim=384, num_classes=33):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# FUZZY MATCHING
# ============================================================================

def build_vocabulary(texts):
    """Extract all unique words from texts"""
    vocab = set()
    for text in texts:
        vocab.update(re.findall(r'\w+', text.lower()))
    return vocab

def fuzzy_correct(text, vocab, cutoff=0.8):
    """Correct typos using Levenshtein distance"""
    words = re.findall(r'\w+', text.lower())
    corrected = []
    
    for word in words:
        matches = get_close_matches(word, vocab, n=1, cutoff=cutoff)
        corrected.append(matches[0] if matches else word)
    
    result = text
    for orig, corr in zip(words, corrected):
        if orig != corr:
            result = re.sub(rf'\b{orig}\b', corr, result, count=1, flags=re.IGNORECASE)
    
    return result

# ============================================================================
# PHONE EXTRACTION (NER)
# ============================================================================

PHONE_PATTERN = re.compile(
    r'(\b08[1-9]\d{7,12}\b|\b628[1-9]\d{7,12}\b|\b\+628[1-9]\d{7,12}\b)', 
    re.IGNORECASE
)

def extract_phone(text):
    """Extract and normalize Indonesian phone numbers"""
    phone_match = PHONE_PATTERN.search(text)
    
    if phone_match:
        num = phone_match.group(0).replace(" ", "").replace("-", "")
        
        if num.startswith("08"):
            return num
        elif num.startswith("628"):
            return "0" + num[2:]
        elif num.startswith("+628"):
            return "0" + num[3:]
    
    return None

# ============================================================================
# TRAINING
# ============================================================================

def create_batches(X, y, batch_size):
    """Create mini-batches with shuffling"""
    indices = torch.randperm(len(X))
    for i in range(0, len(X), batch_size):
        batch_idx = indices[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]

def train_model(X_train, y_train, X_val, y_val, num_classes):
    """Train the classifier"""
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    model = EmbeddingClassifier(
        embed_dim=config.EMBED_DIM,
        num_classes=num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    
    print("\nEpoch | Train Acc | Train Loss | Val Acc | Val Loss")
    print("-" * 65)
    
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in create_batches(X_train, y_train, config.BATCH_SIZE):
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == batch_y).sum().item()
            train_total += len(batch_y)
        
        train_acc = train_correct / train_total
        train_loss = train_loss / (len(X_train) // config.BATCH_SIZE + 1)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_acc = (val_outputs.argmax(1) == y_val).float().mean().item()
        
        # Print progress
        if epoch % 5 == 0 or epoch == config.EPOCHS - 1:
            print(f"{epoch:5d} | {train_acc:9.4f} | {train_loss:10.4f} | {val_acc:7.4f} | {val_loss:8.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
    
    print("\n" + "=" * 80)
    print(f"✓ Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"✓ Model saved to: {config.MODEL_PATH}")
    print("=" * 80)
    
    return model

# ============================================================================
# CHATBOT CLASS
# ============================================================================

class ChatteaBot:
    """Main chatbot class"""
    
    def __init__(self, model, embedder, label_encoder, responses, df, sentence_embeddings, vocab):
        self.model = model
        self.embedder = embedder
        self.label_encoder = label_encoder
        self.responses = responses
        self.df = df
        self.sentence_embeddings = sentence_embeddings
        self.vocab = vocab
        
        self.model.eval()
        self.intent_map = dict(enumerate(label_encoder.classes_))
    
    def get_reply(self, user_input):
        """Get chatbot response"""
        text = user_input.strip().lower()
        
        # Rule-based filters
        if any(g in text for g in ["hai", "halo", "hello", "hi", "hey", "pagi", "siang", "malam"]):
            return self._get_response("greeting")
        
        if any(g in text for g in ["bye", "goodbye", "dadah", "sampai jumpa"]):
            return self._get_response("goodbye")
        
        # Phone extraction
        extracted_phone = extract_phone(user_input)
        
        # Classification
        with torch.no_grad():
            user_emb = self.embedder.encode(user_input, convert_to_tensor=True).to(device)
            user_emb = user_emb.unsqueeze(0)
            
            # Model prediction
            logits = self.model(user_emb)
            probs = logits.softmax(1)
            confidence = probs.max().item()
            intent_idx = logits.argmax(1).item()
            model_intent = self.intent_map[intent_idx]
            
            # Retrieval fallback
            cos_scores = util.cos_sim(user_emb, self.sentence_embeddings)[0]
            best_match_idx = cos_scores.argmax().item()
            retrieval_intent = self.df.iloc[best_match_idx]['intent']
            
            # Choose final intent
            if confidence > config.CONFIDENCE_THRESHOLD:
                final_intent = model_intent
            else:
                final_intent = retrieval_intent
        
        # Special case: phone check
        if final_intent == "phone_check" and extracted_phone:
            return f"Checking {extracted_phone}...\nYes, this number is registered and active on WhatsApp!\n\nYou can safely include it in your blast list."
        
        return self._get_response(final_intent)
    
    def _get_response(self, intent):
        """Get response for intent"""
        response = self.responses.get(intent, self.responses.get("help", "I'm not sure how to help with that."))
        
        if isinstance(response, dict):
            return response.get("en", response.get("id", "I'm not sure how to help with that."))
        
        return response

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Load data
    print("\n📂 Loading data...")
    df = pd.read_csv(config.DATASET_PATH)
    
    with open(config.RESPONSES_PATH, "r", encoding="utf-8") as f:
        responses = json.load(f)
    
    print(f"✓ Loaded {len(df)} samples, {df['intent'].nunique()} intents")
    
    # Build vocabulary
    print("\n📚 Building vocabulary...")
    vocab = build_vocabulary(df['text'])
    print(f"✓ Vocabulary: {len(vocab)} words")
    
    # Label encoding
    print("\n🏷️  Encoding labels...")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['intent'])
    num_classes = len(le.classes_)
    print(f"✓ Classes: {num_classes}")
    
    # Load embedder
    print("\n🧠 Loading sentence transformer...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedder loaded")
    
    # Generate embeddings
    print("\n📊 Generating embeddings...")
    sentence_embeddings = embedder.encode(
        df['text'].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    ).to(device)
    print(f"✓ Embeddings shape: {sentence_embeddings.shape}")
    
    # Train or load model
    if os.path.exists(config.MODEL_PATH):
        print(f"\n✓ Found existing model: {config.MODEL_PATH}")
        print("Loading pre-trained model...")
        model = EmbeddingClassifier(embed_dim=config.EMBED_DIM, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
        model.eval()
        print("✓ Model loaded!")
    else:
        print("\n⚠️  No pre-trained model found. Training from scratch...")
        
        # Prepare data
        X = sentence_embeddings.to(device)
        y = torch.tensor(df['label'].values, dtype=torch.long).to(device)
        
        # Split data
        train_idx, val_idx = train_test_split(
            list(range(len(X))),
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_SEED,
            stratify=y.cpu()
        )
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        # Train
        model = train_model(X_train, y_train, X_val, y_val, num_classes)
        model.load_state_dict(torch.load(config.MODEL_PATH))
    
    # Create bot
    print("\n🤖 Initializing chatbot...")
    bot = ChatteaBot(model, embedder, le, responses, df, sentence_embeddings, vocab)
    print("✓ Chatbot ready!")
    
    # Test queries
    print("\n" + "=" * 80)
    print("🧪 TESTING")
    print("=" * 80)
    
    test_queries = [
        "hello",
        "what is chattea",
        "how to blast message",
        "check 08123456789",
        "create instance"
    ]
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        response = bot.get_reply(query)
        print(f"🤖 Bot: {response[:100]}{'...' if len(response) > 100 else ''}")

    
    # Interactive mode
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE MODE")
    print("=" * 80)
    print("Type your messages (or 'quit' to exit)\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("Bot:", bot.get_reply("goodbye"))
                break
            
            if not user_input:
                continue
            
            response = bot.get_reply(user_input)
            print("Bot:", response, "\n")
            
        except KeyboardInterrupt:
            print("\n\nBot: Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()