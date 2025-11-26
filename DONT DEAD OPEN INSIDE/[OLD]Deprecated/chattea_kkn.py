# Chattea Intent Classifier - KNN with Smart Intent Grouping
# Optimized for YOUR actual dataset (430 samples, 85 intents)
# Strategy: Merge 85 → 25 groups for better performance

"""
INSTALLATION:
pip install sentence-transformers scikit-learn pandas numpy matplotlib seaborn langdetect

DATASET REQUIREMENTS:
- chattea.csv (your 430 samples)
- responses_bilingual.json (your 85 intent responses)
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except:
    LANGDETECT_AVAILABLE = False
    print("⚠️  langdetect not available - using fallback language detection")

print("=" * 80)
print("CHATTEA KNN INTENT CLASSIFIER - OPTIMIZED FOR YOUR DATA")
print("=" * 80)
print("Strategy: Smart Intent Grouping (85 → 25 classes)")
print("=" * 80)

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

class Config:
    # Paths (adjust if needed)
    DATA_CSV = "chattea.csv"
    RESPONSES_JSON = "responses_bilingual.json"
    OUTPUT_DIR = "artifacts_knn"
    
    # Model
    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # KNN parameters (optimized for small dataset)
    N_NEIGHBORS = 3  # Lower for small data
    METRIC = 'cosine'
    WEIGHTS = 'distance'
    
    # Data split
    TEST_SIZE = 0.20
    RANDOM_SEED = 42
    
    # Grouping strategy
    USE_GROUPING = True

config = Config()
Path(config.OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================================
# SECTION 2: CUSTOM INTENT GROUPING (Based on YOUR 85 intents)
# ============================================================================

# Analyzed YOUR actual intents and created optimal groups!
INTENT_GROUPS = {
    # Core user actions (keep separate - important!)
    'greeting': ['greeting'],
    'goodbye': ['goodbye'],
    'gratitude': ['gratitude'],
    'help': ['support_contact'],
    'cancel': ['cancel_action'],
    
    # Account & Auth (2 → 1)
    'account': [
        'account_setup',
        'login_issue'
    ],
    
    # Definition & Info (3 → 1)
    'info': [
        'definition',
        'feature_coming_soon',
        'out_of_scope'  # Map unknown to info request
    ],
    
    # Navigation - All tabs (8 → 1)
    'navigation': [
        'navigation_main_dashboard',
        'navigation_instances_tab',
        'navigation_chat_tab',
        'navigation_grouping_tab',
        'navigation_files_tab',
        'navigation_tasks_tab',
        'navigation_payments_tab'
    ],
    
    # Instance Management (8 → 1)
    'instance': [
        'create_instance',
        'function_create_instance',
        'instance_edit',
        'instance_list',
        'instance_connection_status',
        'instance_logout',
        'instance_delete',
        'instance_view_profile'
    ],
    
    # Pairing/Connection (2 → 1)
    'pairing': [
        'pairing',
        'troubleshoot_qr'
    ],
    
    # Chat & Messaging (4 → 1)
    'chat': [
        'chat_send',
        'chat_read',
        'chat_history',
        'function_send_message'
    ],
    
    # Contacts (4 → 1)
    'contacts': [
        'contacts_manage',
        'contacts_add',
        'contacts_filter',
        'advanced_contact_segmentation'
    ],
    
    # Groups (4 → 1)
    'groups': [
        'group_create',
        'function_create_group',
        'group_export',
        'group_settings_edit'
    ],
    
    # Files (3 → 1)
    'files': [
        'files_upload',
        'files_manage',
        'files_share'
    ],
    
    # Tasks (3 → 1)
    'tasks': [
        'tasks_view',
        'tasks_monitor',
        'tasks_view_failed'
    ],
    
    # Payment (8 → 1)
    'payment': [
        'payment_subscribe',
        'payment_upgrade',
        'payment_status',
        'payment_history',
        'payment_methods',
        'payment_issue',
        'payment_extend'
    ],
    
    # Pricing (3 → 1)
    'pricing': [
        'pricing_query',
        'pricing_currency_idr',
        'pricing_instance_limits'
    ],
    
    # Messaging Operations (6 → 1)
    'messaging': [
        'message_blast',
        'message_blast_status',
        'message_schedule',
        'message_schedule_recurring',
        'message_schedule_timezone'
    ],
    
    # Warmup (2 → 1)
    'warmup': [
        'warmup_enable',
        'warmup_info'
    ],
    
    # API (3 → 1)
    'api': [
        'api_reference',
        'api_send_message',
        'api_webhook_setup'
    ],
    
    # Templates (2 → 1)
    'templates': [
        'templates_create',
        'templates_use'
    ],
    
    # Platform (5 → 1)
    'platform': [
        'platform_compare',
        'platform_cloud_open',
        'platform_desktop_install',
        'platform_desktop_info',
        'platform_updates'
    ],
    
    # Troubleshooting (2 → 1)
    'troubleshoot': [
        'troubleshoot_connection',
        'troubleshoot_qr'  # Also in pairing - will use first match
    ],
    
    # Security (3 → 1)
    'security': [
        'security_privacy',
        'security_payment',
        'security_cloud_data'
    ],
    
    # Tips & Optimization (3 → 1)
    'tips': [
        'general_tips',
        'tips_browser',
        'tips_internet'
    ],
    
    # Special Features (5 → keep separate, important)
    'phone_checker': ['function_check_phone'],
    'analytics': ['analytics_view'],
    'calendar': ['calendar_integration'],
    'auto_reply': ['auto_reply_setup', 'auto_reply_disable'],
    'quota': ['quota_reached'],
}

def create_intent_mapping(groups):
    """Create original_intent → group_name mapping"""
    intent_map = {}
    for group_name, intents in groups.items():
        for intent in intents:
            intent_map[intent] = group_name
    return intent_map

def merge_intents_in_df(df, intent_mapping):
    """Apply intent grouping to dataframe"""
    df = df.copy()
    df['intent_original'] = df['intent']
    df['intent_grouped'] = df['intent'].map(intent_mapping)
    
    # Check for unmapped intents
    unmapped = df[df['intent_grouped'].isna()]
    if len(unmapped) > 0:
        print(f"⚠️  Warning: {len(unmapped)} samples have unmapped intents:")
        print(unmapped['intent'].unique())
        # Map unmapped to 'info' as fallback
        df['intent_grouped'] = df['intent_grouped'].fillna('info')
    
    return df

def create_grouped_responses(original_responses, intent_mapping):
    """Map grouped intents to their responses"""
    grouped_responses = {}
    
    for original_intent, group_name in intent_mapping.items():
        if original_intent in original_responses:
            # Use the first response found for this group
            if group_name not in grouped_responses:
                grouped_responses[group_name] = original_responses[original_intent]
    
    return grouped_responses

# ============================================================================
# SECTION 3: DATA LOADING
# ============================================================================

def load_data():
    """Load and prepare data"""
    print(f"\n📂 Loading data...")
    
    # Load CSV
    df = pd.read_csv(config.DATA_CSV)
    print(f"✓ Loaded {len(df)} samples from CSV")
    print(f"✓ Original intents: {df['intent'].nunique()}")
    
    # Load responses
    with open(config.RESPONSES_JSON, 'r', encoding='utf-8') as f:
        responses = json.load(f)
    print(f"✓ Loaded responses for {len(responses)} intents")
    
    return df, responses

# ============================================================================
# SECTION 4: DATA ANALYSIS
# ============================================================================

def analyze_data(df, title="Dataset Analysis"):
    """Comprehensive data analysis"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}")
    print(f"Avg samples per intent: {len(df) / df['intent'].nunique():.1f}")
    
    # Intent distribution
    intent_counts = df['intent'].value_counts()
    
    print(f"\nIntent Distribution Stats:")
    print(f"  Max: {intent_counts.max()} samples")
    print(f"  Min: {intent_counts.min()} samples")
    print(f"  Median: {intent_counts.median():.0f} samples")
    print(f"  Mean: {intent_counts.mean():.1f} samples")
    
    print(f"\nIntents with < 10 samples:")
    low_sample_intents = intent_counts[intent_counts < 10]
    print(f"  Count: {len(low_sample_intents)} intents")
    if len(low_sample_intents) > 0:
        print(f"  Examples: {list(low_sample_intents.head(10).index)}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Top 20 intents
    intent_counts.head(20).plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Top 20 Intent Distribution')
    axes[0].set_xlabel('Sample Count')
    axes[0].invert_yaxis()
    
    # Sample distribution histogram
    axes[1].hist(intent_counts.values, bins=20, color='coral', edgecolor='black')
    axes[1].set_title('Samples per Intent Distribution')
    axes[1].set_xlabel('Number of Samples')
    axes[1].set_ylabel('Number of Intents')
    axes[1].axvline(intent_counts.mean(), color='red', linestyle='--', label=f'Mean: {intent_counts.mean():.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{config.OUTPUT_DIR}/data_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {config.OUTPUT_DIR}/data_analysis.png")
    plt.close()
    
    return intent_counts

# ============================================================================
# SECTION 5: KNN CLASSIFIER
# ============================================================================

class KNNIntentClassifier:
    """KNN classifier using sentence embeddings"""
    
    def __init__(self, model_name, n_neighbors=3, metric='cosine', weights='distance'):
        print("\n🔧 Initializing KNN Classifier...")
        print(f"   Model: {model_name}")
        print(f"   K-Neighbors: {n_neighbors}")
        print(f"   Metric: {metric}")
        print(f"   Weights: {weights}")
        
        self.embedder = SentenceTransformer(model_name)
        self.knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights
        )
        
        self.train_embeddings = None
        self.is_fitted = False
        
        print("✓ Classifier ready")
    
    def fit(self, texts, labels):
        """Train the classifier"""
        print(f"\n📚 Training on {len(texts)} samples...")
        
        # Encode texts
        print("   Encoding texts...")
        self.train_embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Fit KNN
        print("   Fitting KNN...")
        self.knn.fit(self.train_embeddings, labels)
        self.is_fitted = True
        
        print(f"✓ Training complete! Classes: {len(self.knn.classes_)}")
        return self
    
    def predict_single(self, text):
        """Predict single query with confidence"""
        if not self.is_fitted:
            raise ValueError("Classifier not trained yet!")
        
        # Encode
        embedding = self.embedder.encode([text], convert_to_numpy=True)
        
        # Predict
        intent = self.knn.predict(embedding)[0]
        proba = self.knn.predict_proba(embedding)[0]
        confidence = proba.max()
        
        return intent, confidence
    
    def predict_batch(self, texts):
        """Predict multiple texts"""
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        predictions = self.knn.predict(embeddings)
        probas = self.knn.predict_proba(embeddings)
        confidences = probas.max(axis=1)
        
        return predictions, confidences

# ============================================================================
# SECTION 6: EVALUATION
# ============================================================================

def evaluate_classifier(classifier, test_texts, test_labels):
    """Evaluate with metrics"""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Predict
    predictions, confidences = classifier.predict_batch(test_texts)
    
    # Overall metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted', zero_division=0
    )
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    
    print(f"\n💯 Confidence Statistics:")
    print(f"   Mean:      {confidences.mean():.3f}")
    print(f"   Median:    {np.median(confidences):.3f}")
    print(f"   Min:       {confidences.min():.3f}")
    print(f"   Max:       {confidences.max():.3f}")
    print(f"   > 0.5:     {(confidences > 0.5).sum()} / {len(confidences)} ({(confidences > 0.5).mean()*100:.1f}%)")
    print(f"   > 0.7:     {(confidences > 0.7).sum()} / {len(confidences)} ({(confidences > 0.7).mean()*100:.1f}%)")
    
    # Classification report
    print("\n📋 Per-Class Performance:")
    print(classification_report(test_labels, predictions, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    unique_labels = sorted(set(test_labels) | set(predictions))
    
    if len(unique_labels) <= 30:  # Only plot if reasonable size
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=unique_labels,
            yticklabels=unique_labels,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{config.OUTPUT_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved confusion matrix: {config.OUTPUT_DIR}/confusion_matrix.png")
        plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_confidence': confidences.mean(),
        'median_confidence': np.median(confidences)
    }

# ============================================================================
# SECTION 7: CHATBOT INTERFACE
# ============================================================================

def detect_language(text):
    """Simple language detection"""
    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            return "id" if lang in ["id", "ms"] else "en"
        except:
            pass
    
    # Fallback: simple keyword matching
    id_keywords = {'cara', 'kirim', 'pesan', 'jadwal', 'nomor', 'cek', 'tolong', 'saya', 'bisa', 'gimana'}
    tokens = set(text.lower().split())
    
    if len(tokens & id_keywords) >= 1:
        return "id"
    return "en"

def get_response(intent, responses_dict, language):
    """Get bilingual response"""
    if intent not in responses_dict:
        return "Sorry, I don't have a response for that." if language == "en" else "Maaf, saya tidak punya jawaban untuk itu."
    
    response_entry = responses_dict[intent]
    
    if isinstance(response_entry, dict):
        return response_entry.get(language, response_entry.get('en', response_entry.get('id', '')))
    
    return str(response_entry)

class ChatteaChatbot:
    """Production chatbot"""
    
    def __init__(self, classifier, responses_dict):
        self.classifier = classifier
        self.responses = responses_dict
        print("\n🤖 Chatbot initialized and ready!")
    
    def chat(self, user_input, verbose=True):
        """Process user query"""
        # Predict intent
        intent, confidence = self.classifier.predict_single(user_input)
        
        # Detect language
        language = detect_language(user_input)
        
        # Get response
        response = get_response(intent, self.responses, language)
        
        if verbose:
            print(f"\n🎯 Intent: {intent}")
            print(f"💯 Confidence: {confidence:.3f}")
            print(f"🌍 Language: {language}")
        
        return {
            'intent': intent,
            'confidence': confidence,
            'language': language,
            'response': response
        }

# ============================================================================
# SECTION 8: MAIN PIPELINE
# ============================================================================

def main():
    """Complete pipeline"""
    
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)
    
    df, responses = load_data()
    
    print("\n" + "=" * 80)
    print("STEP 2: ANALYZE ORIGINAL DATA")
    print("=" * 80)
    
    analyze_data(df, "Original Dataset (85 Intents)")
    
    if config.USE_GROUPING:
        print("\n" + "=" * 80)
        print("STEP 3: APPLY INTENT GROUPING")
        print("=" * 80)
        
        intent_mapping = create_intent_mapping(INTENT_GROUPS)
        df = merge_intents_in_df(df, intent_mapping)
        
        print(f"✓ Grouped: 85 → {df['intent_grouped'].nunique()} intents")
        print(f"✓ New avg samples per intent: {len(df) / df['intent_grouped'].nunique():.1f}")
        
        # Use grouped intents
        df['intent'] = df['intent_grouped']
        
        # Create grouped responses
        grouped_responses = create_grouped_responses(responses, intent_mapping)
        print(f"✓ Mapped {len(grouped_responses)} grouped responses")
        
        responses = grouped_responses
        
        analyze_data(df, "After Grouping")
    
    print("\n" + "=" * 80)
    print("STEP 4: SPLIT DATA")
    print("=" * 80)
    
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df['intent'],
        random_state=config.RANDOM_SEED
    )
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    print("\n" + "=" * 80)
    print("STEP 5: TRAIN CLASSIFIER")
    print("=" * 80)
    
    classifier = KNNIntentClassifier(
        model_name=config.MODEL_NAME,
        n_neighbors=config.N_NEIGHBORS,
        metric=config.METRIC,
        weights=config.WEIGHTS
    )
    
    classifier.fit(train_df['text'].tolist(), train_df['intent'].tolist())
    
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATE")
    print("=" * 80)
    
    metrics = evaluate_classifier(
        classifier,
        test_df['text'].tolist(),
        test_df['intent'].tolist()
    )
    
    print("\n" + "=" * 80)
    print("STEP 7: DEMO CHATBOT")
    print("=" * 80)
    
    bot = ChatteaChatbot(classifier, responses)
    
    demo_queries = [
        "What is Chattea?",
        "gimana cara blast message?",
        "how to schedule messages?",
        "cek nomor wa yang valid",
        "tolong bantu saya",
        "what is the pricing plan?",
        "create a new instance",
        "pair with QR code",
        "show me analytics",
        "hello",
    ]
    
    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"👤 User: {query}")
        result = bot.chat(query, verbose=True)
        print(f"🤖 Bot: {result['response'][:150]}{'...' if len(result['response']) > 150 else ''}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.1%}")
    print(f"   F1 Score: {metrics['f1']:.3f}")
    print(f"   Avg Confidence: {metrics['mean_confidence']:.3f}")
    print(f"   Intents: {df['intent'].nunique()} classes")
    print(f"\nArtifacts saved to: {config.OUTPUT_DIR}/")
    
    return classifier, bot, metrics


if __name__ == "__main__":
    classifier, bot, metrics = main()
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("Type your questions (or 'quit' to exit)")
    print("=" * 80)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Goodbye!")
                break
            
            result = bot.chat(user_input, verbose=False)
            print(f"🎯 {result['intent']} ({result['confidence']:.2f})")
            print(f"🤖 {result['response']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")