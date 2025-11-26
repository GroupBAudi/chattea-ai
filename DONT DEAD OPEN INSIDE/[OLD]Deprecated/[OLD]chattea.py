import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class HybridChattea:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.ann_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42
        )
        self.conversation_history = []
        self.message_count = 0
        self.intent_responses = {}
        
    # 1. FUZZY LOGIC
    def fuzzy_match(self, input_text, pattern):
        input_lower = input_text.lower().strip()
        pattern_lower = pattern.lower()
        if input_lower == pattern_lower:
            return 1.0
        if pattern_lower in input_lower or input_lower in pattern_lower:
            return 0.8
        input_chars = set(input_lower)
        pattern_chars = set(pattern_lower)
        if len(input_chars | pattern_chars) == 0:
            return 0
        overlap = len(input_chars & pattern_chars)
        total = len(input_chars | pattern_chars)
        return overlap / total
    
    def fuzzy_preprocess(self, input_text):
        patterns = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon'],
            'help': ['help', 'assist', 'support', 'guide'],
            'create': ['create', 'make', 'add', 'new', 'setup'],
            'view': ['show', 'display', 'view', 'list', 'see'],
            'send': ['send', 'message', 'text', 'broadcast'],
            'check': ['check', 'verify', 'status', 'is'],
            # NEW CATEGORY
            'out_of_scope': ['asdf', 'blah', 'random', 'nonsense', '???', 'idk']
        }

        fuzzy_scores = {}
        for category, pattern_list in patterns.items():
            scores = [self.fuzzy_match(input_text, pattern) for pattern in pattern_list]
            fuzzy_scores[category] = max(scores) if scores else 0

        best_match = max(fuzzy_scores, key=fuzzy_scores.get)
        best_score = fuzzy_scores[best_match]

        # Threshold check: if no good match, force out_of_scope
        if best_score < 0.3:
            return fuzzy_scores, "out_of_scope", best_score

        return fuzzy_scores, best_match, best_score

    
    # 2. ANN
    def train_ann(self, df):
        print("\n" + "="*60)
        print("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
        print("="*60)
        
        X = df['text'].values
        y = df['intent'].values
        print(f"Total samples: {len(X)}")
        print(f"Unique intents: {len(np.unique(y))}")
        
        intent_counts = pd.Series(y).value_counts()
        lonely_intents = intent_counts[intent_counts < 2].index.tolist()
        
        if lonely_intents:
            print(f"\n⚠️  Warning: {len(lonely_intents)} intents have only 1 sample:")
            for intent in lonely_intents:
                print(f"   - {intent}")
            print("   Using non-stratified split to avoid errors.\n")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        print("\nVectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print("Training neural network...")
        print(f"Architecture: Input({X_train_vec.shape[1]}) -> 128 -> 64 -> 32 -> Output({len(np.unique(y))})")
        self.ann_model.fit(X_train_vec, y_train)
        
        y_pred = self.ann_model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[ANN TRAINING COMPLETE]")
        print(f"Training Accuracy: {self.ann_model.score(X_train_vec, y_train):.2%}")
        print(f"Testing Accuracy: {accuracy:.2%}")
        print(f"Model iterations: {self.ann_model.n_iter_}")
        print("="*60 + "\n")
        
        return accuracy
    
    def predict_intent(self, input_text):
        input_vec = self.vectorizer.transform([input_text])
        intent = self.ann_model.predict(input_vec)[0]
        probabilities = self.ann_model.predict_proba(input_vec)[0]
        confidence = max(probabilities)
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_intents = [(self.ann_model.classes_[i], probabilities[i]) for i in top_indices]
        return intent, confidence, top_intents
    
    # 3. EXPERT SYSTEM
    def build_expert_system(self):
        self.intent_responses = {
            'account_setup': "To create an account: Visit Chattea website, click 'Sign Up', enter your details, and verify email.",
            'instance_list': "Fetching your instances... I'll show you all instances with their names, connection status, and phone numbers.",
            'payment_methods': "Accepted payment methods: Credit/Debit Cards, Bank Transfer, E-wallets (GoPay, OVO, DANA), QRIS.",
            'support_contact': "Contact support: Email support@chattea.com or WhatsApp +62xxx.",
            'greeting': "Hello! I'm Chattea Assistant. How can I help you today?",
            'unknown': "I'm not sure I understand. Could you rephrase that?",
            'out_of_scope': "Sorry, I can’t help with that. Try asking about accounts, payments, or messages."
        }
    
    def get_response(self, intent):
        return self.intent_responses.get(
            intent,
            "Sorry, I didn’t understand that. Please ask about accounts, payments, or messages."
        )
    
    # 4. LINEAR REGRESSION
    def linear_regression_quality(self, message_count, avg_length, confidence, fuzzy_score):
        w1, w2, w3, w4, bias = 0.03, 0.001, 0.5, 0.2, 0.25
        quality = (w1 * message_count) + (w2 * avg_length) + (w3 * confidence) + (w4 * fuzzy_score) + bias
        quality = max(0, min(1, quality))
        return quality
    
    def chat(self, user_input):
        print(f"\n{'='*70}")
        print(f"USER: {user_input}")
        print(f"{'='*70}")
        
        self.conversation_history.append({'role': 'user', 'text': user_input})
        self.message_count += 1
        
        fuzzy_scores, best_fuzzy_match, fuzzy_confidence = self.fuzzy_preprocess(user_input)
        print(f"\n[1. FUZZY LOGIC]")
        print(f"  Best pattern match: '{best_fuzzy_match}' (score: {fuzzy_confidence:.2f})")
        print(f"  All scores: {fuzzy_scores}")
        
        intent, confidence, top_intents = self.predict_intent(user_input)
        
        # Apply confidence threshold
        if confidence < 0.70:
            intent = "out_of_scope"
        
        print(f"\n[2. ARTIFICIAL NEURAL NETWORK]")
        print(f"  Predicted Intent: '{intent}' (confidence: {confidence:.2%})")
        print(f"  Top 3 predictions:")
        for i, (int_name, prob) in enumerate(top_intents, 1):
            print(f"    {i}. {int_name}: {prob:.2%}")
        
        response = self.get_response(intent)
        print(f"\n[3. EXPERT SYSTEM]")
        print(f"  Rule triggered for intent: '{intent}'")
        
        total_length = sum(len(msg['text']) for msg in self.conversation_history)
        avg_length = total_length / len(self.conversation_history)
        quality_score = self.linear_regression_quality(
            self.message_count, avg_length, confidence, fuzzy_confidence
        )
        print(f"\n[4. LINEAR REGRESSION]")
        print(f"  Conversation Quality Score: {quality_score:.2%}")
        print(f"  Factors: messages={self.message_count}, avg_length={avg_length:.0f}, ")
        print(f"           ann_confidence={confidence:.2%}, fuzzy_score={fuzzy_confidence:.2f}")
        
        self.conversation_history.append({'role': 'bot', 'text': response})
        
        print(f"\n{'='*70}")
        print(f"BOT: {response}")
        print(f"{'='*70}\n")
        
        return response, intent, confidence, quality_score


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHATTEA HYBRID AI CHATBOT")
    print("Algorithms: Fuzzy Logic + ANN + Expert System + Linear Regression")
    print("="*70)
    
    # Load dataset
    print("\nLoading dataset...")
    df = pd.read_csv('chattea.csv')
    print(f"✓ Loaded {len(df)} training examples")
    print(f"✓ Found {df['intent'].nunique()} unique intents")
    
    # Initialize chatbot
    chatbot = HybridChattea()
    
    # Build expert system
    print("✓ Building Expert System rules...")
    chatbot.build_expert_system()
    
    # Train ANN
    accuracy = chatbot.train_ann(df)
    
    # Start conversation
    print("\n" + "="*70)
    print("CHATBOT READY! Type 'quit', 'exit', or 'bye' to end conversation")
    print("="*70)
    print("\nTry asking:")
    print("  - 'What is Chattea?'")
    print("  - 'How do I create an instance?'")
    print("  - 'Send message to +628123456789'")
    print("  - 'What are the pricing plans?'")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\nBot: Goodbye! Thanks for using Chattea Assistant. Have a great day! 👋")
                break
            
            chatbot.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\nBot: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            print("Please try again or type 'quit' to exit.")