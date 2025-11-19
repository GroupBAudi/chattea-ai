import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Hyperparameters and settings"""
    # Paths
    DATA_PATH = 'chattea.csv'
    MODEL_SAVE_PATH = 'chattea_model_gpu.pth'
    RESPONSES_PATH = 'responses.json'
    
    # Model architecture
    EMBEDDING_DIM = 256      # Larger embedding (we have GPU power!)
    HIDDEN_DIM = 512         # Larger hidden layer
    MAX_SEQ_LEN = 40         # Max words per query
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 64          # Larger batch (GPU can handle it)
    EPOCHS = 50              # More epochs for better convergence
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5      # L2 regularization
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data split
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Preprocessing
    MIN_FREQ = 1             # Include all words (we have enough data)
    
    # Inference
    CONFIDENCE_THRESHOLD = 0.65

print(f"🚀 Using device: {Config.DEVICE}")
if Config.DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# PART 1: CUSTOM PREPROCESSING
# ============================================================================

class IntentPreprocessor:
    """
    Enhanced preprocessor for bilingual dataset (English + Indonesian)
    """
    
    def __init__(self, max_len=Config.MAX_SEQ_LEN):
        self.max_len = max_len
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word2idx = self.vocab.copy()
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
        
    def tokenize(self, text):
        """
        Tokenization supporting both English and Indonesian.
        Handles punctuation, numbers, and mixed case.
        """
        if not isinstance(text, str):
            return []
        
        # Lowercase
        text = text.lower()
        
        # Keep apostrophes but remove other punctuation
        text = re.sub(r"[^a-z0-9\s']", ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split
        tokens = text.split()
        
        return tokens
    
    def build_vocab(self, texts, min_freq=Config.MIN_FREQ):
        """Build vocabulary from texts"""
        word_freq = Counter()
        
        print("📊 Building vocabulary...")
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = self.tokenize(text)
            word_freq.update(tokens)
        
        # Add words to vocabulary
        idx = 2
        for word, freq in word_freq.items():
            if freq >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        self.vocab_size = len(self.word2idx)
        
        print(f"✓ Vocabulary built:")
        print(f"  - Total unique words: {len(word_freq)}")
        print(f"  - Vocab size (min_freq={min_freq}): {self.vocab_size}")
        print(f"  - Most common words: {word_freq.most_common(10)}")
        
        return self
    
    def encode(self, text):
        """Convert text to padded sequence"""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, 1) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return indices
    
    def encode_batch(self, texts):
        """Encode multiple texts"""
        return [self.encode(text) for text in texts]
    
    def decode(self, indices):
        """Convert indices back to text"""
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in indices if idx != 0]
        return ' '.join(tokens)

# ============================================================================
# PART 2: ENHANCED MODEL ARCHITECTURE
# ============================================================================

class EnhancedAttentionClassifier(nn.Module):
    """
    Enhanced neural network with:
    - Bidirectional attention
    - Residual connections
    - Layer normalization
    - Multiple attention heads (simplified multi-head attention)
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.3):
        super(EnhancedAttentionClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Attention mechanism (2 heads for richer representation)
        self.num_heads = 2
        head_dim = embedding_dim // self.num_heads
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.attention_scale = np.sqrt(head_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network (deeper)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, x):
        """
        Forward pass with multi-head attention.
        
        Args:
            x: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, seq_len = x.size()
        
        # 1. Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # 2. Multi-head attention
        Q = self.query(embedded)  # (batch_size, seq_len, embedding_dim)
        K = self.key(embedded)
        V = self.value(embedded)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_scale
        
        # Mask padding
        padding_mask = (x == 0).unsqueeze(1).unsqueeze(2)
        attention_scores = attention_scores.masked_fill(padding_mask, -1e9)
        
        # Attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Residual connection + Layer norm
        attended = self.layer_norm1(attended + embedded)
        
        # 3. Pooling (mean over sequence, ignoring padding)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # 4. Deep feed-forward network
        x = self.relu(self.fc1(pooled))
        x = self.layer_norm2(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        
        logits = self.fc4(x)
        
        return logits

# ============================================================================
# PART 3: DATASET CLASS
# ============================================================================

class IntentDataset(Dataset):
    """PyTorch Dataset"""
    
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.encoded = preprocessor.encode_batch(texts)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encoded[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ============================================================================
# PART 4: GPU-OPTIMIZED TRAINING
# ============================================================================

class GPUTrainer:
    """
    GPU-optimized training pipeline with:
    - Mixed precision training (optional)
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    """
    
    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward
            optimizer.zero_grad()
            logits = self.model(input_ids)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion):
        """Validate"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(input_ids)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=Config.EPOCHS, lr=Config.LEARNING_RATE):
        """Full training with early stopping"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        print("\n" + "="*70)
        print("🚀 TRAINING STARTED")
        print("="*70)
        
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(val_loader, criterion)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Print
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }, Config.MODEL_SAVE_PATH)
                print(f"   ✓ Best model saved! (Acc: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
                break
        
        print(f"\n✅ Training complete! Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Load best model
        checkpoint = torch.load(Config.MODEL_SAVE_PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def plot_training(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(self.train_losses) + 1)
        
        # Loss
        ax1.plot(epochs_range, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs_range, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs_range, self.val_accuracies, 'g-', linewidth=2)
        ax2.axhline(y=max(self.val_accuracies), color='r', linestyle='--', 
                    label=f'Best: {max(self.val_accuracies):.4f}')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves_gpu.png', dpi=300, bbox_inches='tight')
        print("✓ Training curves saved to 'training_curves_gpu.png'")
        plt.close()
    
    def evaluate(self, dataloader, intent_names):
        """Detailed evaluation"""
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, preds, labels = self.validate(dataloader, criterion)
        
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        print(f"Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(labels, preds, target_names=intent_names, digits=4))
        
        # Confusion matrix (only for smaller subset if too many classes)
        if len(intent_names) <= 30:
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(20, 16))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=intent_names,
                        yticklabels=intent_names,
                        cbar_kws={'label': 'Count'})
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.ylabel('True Label', fontsize=14)
            plt.xlabel('Predicted Label', fontsize=14)
            plt.xticks(rotation=90, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('confusion_matrix_gpu.png', dpi=300, bbox_inches='tight')
            print("\n✓ Confusion matrix saved to 'confusion_matrix_gpu.png'")
            plt.close()
        else:
            print(f"\n⚠️  Skipping confusion matrix visualization ({len(intent_names)} classes too many)")

# ============================================================================
# PART 5: INFERENCE WITH RESPONSES
# ============================================================================

class ChatteaBot:
    """Production chatbot"""
    
    def __init__(self, model, preprocessor, intent_map, responses_dict, 
                 confidence_threshold=Config.CONFIDENCE_THRESHOLD, device=Config.DEVICE):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.intent_map = intent_map
        self.responses = responses_dict
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model.eval()
    
    def predict(self, text):
        """Predict with confidence"""
        # Encode
        encoded = self.preprocessor.encode(text)
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        # Get intent
        intent = self.intent_map[predicted_idx.item()]
        conf = confidence.item()
        
        # Get response
        if conf < self.confidence_threshold:
            response = self.responses.get('out_of_scope', 
                "Maaf, saya kurang mengerti. Bisa tolong diulang dengan cara lain?")
            intent = "out_of_scope_low_confidence"
        else:
            response = self.responses.get(intent, 
                f"Intent '{intent}' recognized but no response configured.")
        
        return {
            'intent': intent,
            'confidence': conf,
            'response': response
        }
    
    def chat(self):
        """Interactive chat"""
        print("\n" + "="*70)
        print("💬 CHATTEA BOT - Interactive Mode")
        print("="*70)
        print("Ketik 'exit' untuk keluar\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'keluar']:
                print("Bot: Terima kasih! Sampai jumpa! 👋")
                break
            
            if not user_input:
                continue
            
            result = self.predict(user_input)
            print(f"Bot: {result['response']}")
            print(f"     [Intent: {result['intent']}, Confidence: {result['confidence']:.3f}]\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🤖 CHATTEA INTENT CLASSIFIER - GPU TRAINING PIPELINE")
    print("="*70)
    
    # 1. Load data
    print("\n📂 Loading dataset...")
    df = pd.read_csv(Config.DATA_PATH)
    
    print(f"✓ Dataset loaded:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Unique intents: {df['intent'].nunique()}")
    print(f"  - Intent distribution:\n{df['intent'].value_counts().head(10)}")
    
    # Prepare data
    texts = df['text'].tolist()
    intents = df['intent'].tolist()
    
    # Create intent mapping
    unique_intents = sorted(df['intent'].unique())
    intent_to_idx = {intent: idx for idx, intent in enumerate(unique_intents)}
    idx_to_intent = {idx: intent for intent, idx in intent_to_idx.items()}
    
    labels = [intent_to_idx[intent] for intent in intents]
    
    # 2. Split data
    print("\n✂️  Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=Config.VAL_SIZE/(1-Config.TEST_SIZE), 
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"✓ Data split:")
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Val:   {len(X_val)} samples")
    print(f"  - Test:  {len(X_test)} samples")
    
    # 3. Build preprocessor
    preprocessor = IntentPreprocessor()
    preprocessor.build_vocab(X_train)
    
    # 4. Create datasets
    train_dataset = IntentDataset(X_train, y_train, preprocessor)
    val_dataset = IntentDataset(X_val, y_val, preprocessor)
    test_dataset = IntentDataset(X_test, y_test, preprocessor)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 5. Initialize model
    print("\n🧠 Initializing model...")
    model = EnhancedAttentionClassifier(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_classes=len(unique_intents),
        dropout=Config.DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model initialized:")
    print(f"  - Architecture: EnhancedAttentionClassifier")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: ~{total_params * 4 / 1e6:.2f} MB")
    
    # 6. Train
    trainer = GPUTrainer(model)
    trainer.train(train_loader, val_loader)
    
    # 7. Plot
    trainer.plot_training()
    
    # 8. Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    trainer.evaluate(test_loader, unique_intents)
    
    # 9. Save everything
    print("\n💾 Saving artifacts...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': preprocessor,
        'intent_map': idx_to_intent,
        'config': {
            'vocab_size': preprocessor.vocab_size,
            'embedding_dim': Config.EMBEDDING_DIM,
            'hidden_dim': Config.HIDDEN_DIM,
            'num_classes': len(unique_intents),
            'max_seq_len': Config.MAX_SEQ_LEN
        }
    }, 'chattea_complete_model.pth')
    
    print("✓ Model saved to 'chattea_complete_model.pth'")
    
    # 10. Create simple responses (you should customize these)
    responses = {intent: f"Response for {intent}" for intent in unique_intents}
    responses['out_of_scope'] = "Maaf, saya kurang mengerti. Bisa tolong diulang?"
    
    with open(Config.RESPONSES_PATH, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Responses template saved to '{Config.RESPONSES_PATH}'")
    
    # 11. Test interactive mode
    print("\n" + "="*70)
    print("🧪 TESTING MODEL")
    print("="*70)
    
    bot = ChatteaBot(model, preprocessor, idx_to_intent, responses)
    
    # Test queries
    test_queries = [
        "How do I create an account?",
        "send bulk message",
        "what is chattea",
        "asdfasdfasdf",  # Out of scope
        "help me with payment"
    ]
    
    print("\nTest predictions:")
    for query in test_queries:
        result = bot.predict(query)
        print(f"\nQuery: {query}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Response: {result['response'][:80]}...")
    
    # Start interactive chat
    print("\n" + "="*70)
    user_choice = input("\n💬 Start interactive chat? (y/n): ").strip().lower()
    if user_choice == 'y':
        bot.chat()

if __name__ == "__main__":
    main()