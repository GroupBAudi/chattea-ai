import pandas as pd
import time
from googletrans import Translator
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    INPUT_CSV = 'chattea.csv'
    OUTPUT_CSV = 'chattea_bilingual.csv'
    BATCH_SIZE = 50  # Translate in batches to avoid rate limits
    DELAY_SECONDS = 1  # Delay between batches
    
    # Manual corrections for common translation issues
    MANUAL_FIXES = {
        # Common tech terms that shouldn't be translated
        'WhatsApp': 'WhatsApp',
        'Chattea': 'Chattea',
        'QR code': 'kode QR',
        'API': 'API',
        'webhook': 'webhook',
        'instance': 'instance',
        'warmup': 'warmup',
        'blast': 'blast',
        'dashboard': 'dashboard',
        'template': 'template',
        
        # Common phrase corrections
        'how do i': 'bagaimana cara',
        'how to': 'cara',
        'i want to': 'saya ingin',
        'i need to': 'saya perlu',
        'can you': 'bisakah anda',
        'help me': 'bantu saya',
        'show me': 'tampilkan',
        'please': 'tolong',
        'thank you': 'terima kasih',
        'thanks': 'terima kasih',
    }

# ============================================================================
# INTELLIGENT TRANSLATOR
# ============================================================================

class IntelligentTranslator:
    """
    Translates with context awareness and manual corrections.
    """
    
    def __init__(self):
        self.translator = Translator()
        self.translation_cache = {}
        self.failed_translations = []
        
    def translate_text(self, text, intent=''):
        """
        Translate with intelligent corrections.
        
        Args:
            text: English text to translate
            intent: Intent category for context
        
        Returns:
            Translated Indonesian text
        """
        if not isinstance(text, str) or not text.strip():
            return text
        
        # Check cache
        cache_key = text.lower()
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Translate
            translated = self.translator.translate(text, src='en', dest='id').text
            
            # Apply manual fixes
            translated = self._apply_manual_fixes(translated, text)
            
            # Cache result
            self.translation_cache[cache_key] = translated
            
            return translated
            
        except Exception as e:
            print(f"⚠️  Translation failed for: '{text}' - Error: {e}")
            self.failed_translations.append(text)
            return text  # Return original if translation fails
    
    def _apply_manual_fixes(self, translated, original):
        """Apply manual corrections to translation"""
        # Preserve proper nouns
        if 'Chattea' in original and 'Chattea' not in translated:
            translated = translated.replace('chattea', 'Chattea').replace('Chatya', 'Chattea')
        
        # Fix common tech terms
        replacements = {
            'Kod QR': 'kode QR',
            'kod qr': 'kode QR',
            'perpanjangan web': 'webhook',
            'contoh': 'instance',
            'pemanasan': 'warmup',
            'ledakan': 'blast',
            'papan pemukul': 'dashboard',
        }
        
        for wrong, correct in replacements.items():
            if wrong in translated:
                translated = translated.replace(wrong, correct)
        
        return translated
    
    def translate_batch(self, texts, intent=''):
        """Translate a batch of texts"""
        results = []
        for text in texts:
            translated = self.translate_text(text, intent)
            results.append(translated)
            time.sleep(0.1)  # Small delay to avoid rate limits
        return results

# ============================================================================
# DATASET GENERATOR
# ============================================================================

class BilingualDatasetGenerator:
    """
    Generates bilingual dataset with quality controls.
    """
    
    def __init__(self, input_csv):
        self.df = pd.read_csv(input_csv)
        self.translator = IntelligentTranslator()
        
    def generate_bilingual_dataset(self, output_csv, sample_ratio=1.0):
        """
        Generate bilingual dataset.
        
        Args:
            output_csv: Output file path
            sample_ratio: Ratio of data to translate (1.0 = 100%)
        """
        print("\n" + "="*70)
        print("🌍 BILINGUAL DATASET GENERATOR")
        print("="*70)
        print(f"📂 Input: {Config.INPUT_CSV}")
        print(f"📊 Total samples: {len(self.df)}")
        print(f"🎯 Unique intents: {self.df['intent'].nunique()}")
        
        # Sample if needed
        if sample_ratio < 1.0:
            sample_size = int(len(self.df) * sample_ratio)
            df_sample = self.df.sample(n=sample_size, random_state=42)
            print(f"🎲 Sampling {sample_size} samples ({sample_ratio*100:.0f}%)")
        else:
            df_sample = self.df
        
        # Translate in batches
        total_samples = len(df_sample)
        indonesian_data = []
        
        print(f"\n🔄 Translating {total_samples} samples to Indonesian...")
        print("This may take 5-10 minutes...\n")
        
        for i in range(0, total_samples, Config.BATCH_SIZE):
            batch_end = min(i + Config.BATCH_SIZE, total_samples)
            batch = df_sample.iloc[i:batch_end]
            
            print(f"📦 Batch {i//Config.BATCH_SIZE + 1}/{(total_samples//Config.BATCH_SIZE)+1} "
                  f"({i+1}-{batch_end}/{total_samples})")
            
            # Translate batch
            for _, row in batch.iterrows():
                intent = row['intent']
                
                # Skip out_of_scope (keep original English nonsense)
                if intent == 'out_of_scope':
                    continue
                
                translated_text = self.translator.translate_text(row['text'], intent)
                
                indonesian_data.append({
                    'text': translated_text,
                    'intent': intent
                })
            
            # Progress indicator
            progress = (batch_end / total_samples) * 100
            print(f"   ✓ Progress: {progress:.1f}%")
            
            # Delay between batches
            if batch_end < total_samples:
                time.sleep(Config.DELAY_SECONDS)
        
        # Create bilingual dataframe
        df_indonesian = pd.DataFrame(indonesian_data)
        
        # Combine English + Indonesian
        df_bilingual = pd.concat([self.df, df_indonesian], ignore_index=True)
        
        # Shuffle
        df_bilingual = df_bilingual.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        df_bilingual.to_csv(output_csv, index=False, encoding='utf-8')
        
        # Print statistics
        print("\n" + "="*70)
        print("✅ TRANSLATION COMPLETE!")
        print("="*70)
        print(f"📊 Statistics:")
        print(f"  - Original English samples: {len(self.df)}")
        print(f"  - Indonesian translations: {len(df_indonesian)}")
        print(f"  - Total bilingual dataset: {len(df_bilingual)}")
        print(f"  - Language distribution:")
        
        # Approximate language detection
        en_count = len(self.df)
        id_count = len(df_indonesian)
        print(f"    • English: {en_count} ({en_count/len(df_bilingual)*100:.1f}%)")
        print(f"    • Indonesian: {id_count} ({id_count/len(df_bilingual)*100:.1f}%)")
        
        print(f"\n💾 Saved to: {output_csv}")
        
        if self.translator.failed_translations:
            print(f"\n⚠️  {len(self.translator.failed_translations)} translations failed:")
            for failed in self.translator.failed_translations[:10]:
                print(f"   - {failed}")
        
        # Show sample translations
        print("\n📝 Sample translations:")
        samples = df_indonesian.sample(min(10, len(df_indonesian)))
        for _, row in samples.iterrows():
            print(f"\n  Intent: {row['intent']}")
            print(f"  ID: {row['text']}")
    
    def validate_translations(self, bilingual_csv):
        """
        Validate translation quality by showing random samples.
        """
        df = pd.read_csv(bilingual_csv)
        
        print("\n" + "="*70)
        print("🔍 TRANSLATION VALIDATION")
        print("="*70)
        
        # Group by intent and show EN/ID pairs
        intents = df['intent'].unique()[:5]  # Check first 5 intents
        
        for intent in intents:
            print(f"\n📌 Intent: {intent}")
            samples = df[df['intent'] == intent].head(4)
            
            for i, (_, row) in enumerate(samples.iterrows(), 1):
                # Try to determine language
                has_indonesian = any(word in row['text'].lower() for word in 
                                    ['bagaimana', 'cara', 'saya', 'untuk', 'dengan'])
                lang = 'ID' if has_indonesian else 'EN'
                print(f"  {i}. [{lang}] {row['text']}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate bilingual dataset"""
    
    try:
        # Initialize generator
        generator = BilingualDatasetGenerator(Config.INPUT_CSV)
        
        # Generate bilingual dataset
        generator.generate_bilingual_dataset(
            output_csv=Config.OUTPUT_CSV,
            sample_ratio=1.0  # Translate 100% of data
        )
        
        # Validate translations
        generator.validate_translations(Config.OUTPUT_CSV)
        
        print("\n" + "="*70)
        print("🎉 SUCCESS!")
        print("="*70)
        print(f"\n✅ Your bilingual dataset is ready: {Config.OUTPUT_CSV}")
        print("\n🚀 Next steps:")
        print("  1. Manually review a few translations (optional)")
        print("  2. Update your training script to use 'chattea_bilingual.csv'")
        print("  3. Retrain your model")
        print("  4. Test with Indonesian queries!")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{Config.INPUT_CSV}'")
        print("   Make sure the file exists in the current directory.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()