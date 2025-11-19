import pandas as pd
import numpy as np
import random
import re

# Load dataset
df = pd.read_csv('chattea.csv')

print("="*70)
print("CHATTEA NATURAL DATASET AUGMENTATION - TARGET: 30+ SAMPLES")
print("="*70)
print(f"\nOriginal dataset:")
print(f"  Total samples: {len(df)}")
print(f"  Unique intents: {df['intent'].nunique()}")

intent_counts = df['intent'].value_counts()
needs_augmentation = intent_counts[intent_counts < 30]

print(f"\nIntents needing augmentation (< 30 samples): {len(needs_augmentation)}")

TARGET_SAMPLES = 30

class NaturalAugmenter:
    """Generate natural-sounding variations"""
    
    def __init__(self):
        # Question starters (natural phrasing)
        self.question_starters = [
            "How do I", "How can I", "How to", "Can you help me",
            "Could you show me how to", "What's the best way to",
            "I need help with", "Can you tell me how to"
        ]
        
        # Command starters
        self.command_starters = [
            "Please", "I want to", "I need to", "I'd like to",
            "Help me", "Can you", "Could you"
        ]
        
        # Polite endings (use sparingly)
        self.polite_endings = ["please", "thanks", "thank you", ""]
        
        # Synonym mappings (only for key verbs)
        self.verb_synonyms = {
            'show': ['display', 'view'],
            'open': ['access', 'go to'],
            'create': ['make', 'add', 'set up'],
            'send': ['deliver', 'transmit'],
            'check': ['verify', 'confirm'],
            'delete': ['remove', 'erase'],
            'list': ['show all', 'display all'],
        }
    
    def is_question(self, text):
        """Check if text is a question"""
        return text.strip().endswith('?') or any(text.lower().startswith(q) for q in ['how', 'what', 'where', 'when', 'why', 'who', 'can', 'is', 'do', 'does'])
    
    def is_command(self, text):
        """Check if text is a command"""
        command_starts = ['show', 'open', 'create', 'send', 'check', 'delete', 'go', 'view', 'list', 'display']
        return any(text.lower().startswith(cmd) for cmd in command_starts)
    
    def clean_text(self, text):
        """Remove artifacts from text"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Fix spacing around punctuation
        text = re.sub(r'\s+([?.!,])', r'\1', text)
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
    
    def replace_verb(self, text):
        """Replace main verb with synonym"""
        variations = []
        text_lower = text.lower()
        
        for verb, synonyms in self.verb_synonyms.items():
            # Check if verb appears as whole word
            pattern = r'\b' + verb + r'\b'
            if re.search(pattern, text_lower):
                for syn in synonyms:
                    new_text = re.sub(pattern, syn, text_lower)
                    variations.append(self.clean_text(new_text))
        
        return variations
    
    def add_question_variation(self, text):
        """Add question variations"""
        variations = []
        
        # Remove existing question words
        cleaned = re.sub(r'^(how do i|how can i|how to|what|where|when)\s+', '', text.lower(), flags=re.IGNORECASE)
        cleaned = cleaned.rstrip('?').strip()
        
        # Add different question starters
        for starter in random.sample(self.question_starters, min(3, len(self.question_starters))):
            new_text = f"{starter} {cleaned}?"
            variations.append(self.clean_text(new_text))
        
        return variations
    
    def add_command_variation(self, text):
        """Add command variations"""
        variations = []
        
        # Get base command (remove command starters)
        cleaned = text.lower()
        for starter in ['please', 'i want to', 'i need to', 'help me', 'can you', 'could you']:
            cleaned = re.sub(r'^' + starter + r'\s+', '', cleaned, flags=re.IGNORECASE)
        
        cleaned = cleaned.strip()
        
        # Add different command starters
        for starter in random.sample(self.command_starters, min(3, len(self.command_starters))):
            if starter.lower() in ['please']:
                new_text = f"{cleaned.capitalize()} {starter}"
            else:
                new_text = f"{starter} {cleaned}"
            
            # Randomly add polite ending
            if random.random() < 0.3:  # 30% chance
                ending = random.choice([e for e in self.polite_endings if e])
                if ending:
                    new_text = f"{new_text} {ending}"
            
            variations.append(self.clean_text(new_text))
        
        return variations
    
    def add_casual_variations(self, text):
        """Add more casual/informal variations"""
        variations = []
        
        # Contractions
        contractions = {
            'I would like': "I'd like",
            'I am': "I'm",
            'cannot': "can't",
            'do not': "don't",
            'what is': "what's",
        }
        
        for formal, casual in contractions.items():
            if formal.lower() in text.lower():
                new_text = text.replace(formal, casual).replace(formal.capitalize(), casual.capitalize())
                variations.append(self.clean_text(new_text))
        
        return variations
    
    def generate_variations(self, text, target_count=10):
        """Generate natural variations of text"""
        if not text or len(text.strip()) < 3:
            return [text]
        
        variations = [text]  # Keep original
        
        # Generate based on text type
        if self.is_question(text):
            variations.extend(self.add_question_variation(text))
        elif self.is_command(text):
            variations.extend(self.add_command_variation(text))
        else:
            # Mixed approach
            variations.extend(self.add_question_variation(text))
            variations.extend(self.add_command_variation(text))
        
        # Add verb synonyms
        variations.extend(self.replace_verb(text))
        
        # Add casual variations
        variations.extend(self.add_casual_variations(text))
        
        # Clean and deduplicate
        variations = [self.clean_text(v) for v in variations]
        variations = [v for v in variations if v and len(v) > 3]  # Remove empty/too short
        variations = list(dict.fromkeys(variations))  # Remove duplicates, preserve order
        
        # If we need more, slightly modify existing ones
        while len(variations) < target_count:
            base = random.choice(variations[:5])  # Pick from original variations
            
            # Add polite word
            if random.random() < 0.5:
                ending = random.choice([e for e in self.polite_endings if e])
                if ending and ending not in base.lower():
                    new_var = f"{base.rstrip('.!?')} {ending}"
                    new_var = self.clean_text(new_var)
                    if new_var not in variations:
                        variations.append(new_var)
            else:
                # Slight rephrasing
                if '?' in base and random.random() < 0.5:
                    new_var = base.replace('?', '.')
                elif '.' in base and random.random() < 0.5:
                    new_var = base.replace('.', '?')
                else:
                    new_var = base.rstrip('.!?')  # Remove punctuation
                
                new_var = self.clean_text(new_var)
                if new_var not in variations and new_var != base:
                    variations.append(new_var)
        
        return variations[:target_count]

# High-quality hand-crafted data for common intents
handcrafted_data = {
    'greeting': [
        "Hi", "Hello", "Hey", "Hi there", "Hello there", "Hey there",
        "Good morning", "Good afternoon", "Good evening", "Greetings",
        "Yo", "What's up", "Howdy", "Hi!", "Hello!", "Hey!",
        "Morning", "Afternoon", "Evening", "Hi chatbot", "Hello bot"
    ],
    'goodbye': [
        "Bye", "Goodbye", "See you", "See ya", "Later", "Bye bye",
        "Farewell", "Take care", "Catch you later", "Until next time",
        "Cya", "Bye for now", "See you later", "Talk to you later",
        "Gotta go", "Peace", "Adios", "Cheerio", "Bye now"
    ],
    'gratitude': [
        "Thanks", "Thank you", "Thanks!", "Thank you!", "Thx", "Ty",
        "Appreciate it", "Thanks a lot", "Thank you so much",
        "Much appreciated", "Thanks so much", "Thanks a bunch",
        "Many thanks", "Cheers", "Thanks for the help",
        "Thank you for your help", "Thanks mate", "Grateful"
    ],
    'cancel_action': [
        "Cancel", "Stop", "Never mind", "Nevermind", "Forget it",
        "Abort", "Cancel that", "Stop that", "Don't do that",
        "Ignore that", "Skip", "Skip it", "Disregard", "No thanks",
        "Not now", "Maybe later", "I changed my mind"
    ],
    'unknown': [
        "asdf", "qwerty", "xyz", "???", "...", "huh", "what",
        "idk", "random text", "gibberish", "nonsense", "blah blah",
        "test", "123", "abc", "zzz", "hmm", "whatever"
    ],
}

# Initialize augmenter
augmenter = NaturalAugmenter()

# Collect augmented data
augmented_rows = []

for intent in intent_counts.index:
    current_samples = df[df['intent'] == intent]
    current_count = len(current_samples)
    needed = max(0, TARGET_SAMPLES - current_count)
    
    if needed == 0:
        continue
    
    print(f"  Augmenting '{intent}': {current_count} → {TARGET_SAMPLES} (+{needed})")
    
    # Use handcrafted data if available
    if intent in handcrafted_data:
        handcrafted = handcrafted_data[intent]
        for i in range(needed):
            text = handcrafted[i % len(handcrafted)]
            augmented_rows.append({'text': text, 'intent': intent})
    else:
        # Generate natural variations
        all_variations = []
        
        for _, row in current_samples.iterrows():
            original_text = row['text']
            variations = augmenter.generate_variations(original_text, target_count=15)
            all_variations.extend(variations)
        
        # Remove original texts
        original_texts = set(current_samples['text'].tolist())
        all_variations = [v for v in all_variations if v not in original_texts]
        
        # Deduplicate
        all_variations = list(dict.fromkeys(all_variations))
        
        # Shuffle and take what we need
        random.shuffle(all_variations)
        for var in all_variations[:needed]:
            augmented_rows.append({'text': var, 'intent': intent})
        
        # If still short, repeat best variations
        if len(all_variations) < needed:
            shortfall = needed - len(all_variations)
            print(f"    ⚠️  Generated {len(all_variations)}/{needed}, repeating best samples")
            
            for i in range(shortfall):
                if all_variations:
                    text = all_variations[i % len(all_variations)]
                else:
                    text = current_samples.iloc[i % len(current_samples)]['text']
                augmented_rows.append({'text': text, 'intent': intent})

# Create augmented DataFrame
df_augmented = pd.DataFrame(augmented_rows)

# Combine with original
df_final = pd.concat([df, df_augmented], ignore_index=True)

# Remove exact duplicates
before_dedup = len(df_final)
df_final = df_final.drop_duplicates(subset=['text'], keep='first')
after_dedup = len(df_final)

print("\n" + "="*70)
print("AUGMENTATION COMPLETE")
print("="*70)
print(f"\nStatistics:")
print(f"  Original samples: {len(df)}")
print(f"  Generated samples: {len(df_augmented)}")
print(f"  Duplicates removed: {before_dedup - after_dedup}")
print(f"  Final total: {len(df_final)}")

# Check distribution
new_counts = df_final['intent'].value_counts()
below_target = new_counts[new_counts < TARGET_SAMPLES]

print(f"\n  Intents below target ({TARGET_SAMPLES}): {len(below_target)}")
if len(below_target) > 0:
    print(f"  Showing first 10:")
    for intent, count in below_target.head(10).items():
        print(f"    - {intent}: {count} samples")

print(f"\n  Distribution:")
print(f"    Mean: {new_counts.mean():.1f} samples/intent")
print(f"    Median: {new_counts.median():.1f}")
print(f"    Min: {new_counts.min()}")
print(f"    Max: {new_counts.max()}")

# Save
output_file = 'chattea_natural.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✓ Clean dataset saved as '{output_file}'")
print("="*70)

# Show samples of augmented data
print("\n📋 Sample of NEW augmented data:")
sample_intent = random.choice(needs_augmentation.index)
sample_data = df_final[df_final['intent'] == sample_intent].tail(10)
print(f"\nIntent: {sample_intent}")
for text in sample_data['text']:
    print(f"  - {text}")

print("\n✅ Dataset is now clean and natural!")
print("💡 Use 'chattea_natural.csv' in your chatbot training")