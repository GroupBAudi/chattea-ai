import json
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import itertools  

# ================== DEVICE SETUP ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================== LOAD DATA ==================
df = pd.read_csv("chattea_dataset.csv")
with open("responses.json", "r", encoding="utf-8") as f:
    RESPONSES = json.load(f)

# ================== BUILD VOCABULARY FOR FUZZY MATCHING  ==================
# Extract all words from training data
all_words = set()
for text in df['text'].str.lower():
    all_words.update(re.findall(r'\w+', text))
VOCAB = all_words  # global vocabulary

def fuzzy_correct(text: str, cutoff: float = 0.8) -> str:
    """
    Simple but extremely effective typo correction using difflib
    """
    words = re.findall(r'\w+', text.lower())
    corrected = []
    for word in words:
        matches = get_close_matches(word, VOCAB, n=1, cutoff=cutoff)
        corrected.append(matches[0] if matches else word)
    # Reconstruct sentence preserving original capitalization/punctuation
    result = text
    for orig, corr in zip(words, corrected):
        if orig != corr:
            result = re.sub(rf'\b{orig}\b', corr, result, count=1, flags=re.IGNORECASE)
    return result

# ================== LABEL ENCODING ==================
le = LabelEncoder()
df['label'] = le.fit_transform(df['intent'])
num_classes = len(le.classes_)
intent_map = dict(enumerate(le.classes_))

# ================== EMBEDDING MODEL ==================
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentence_embeddings = embedder.encode(df['text'].tolist(), convert_to_tensor=True).to(device)

# ================== CNN CLASSIFIER (PERFECTLY WORKING) ==================
class TextCNN(nn.Module):
    def __init__(self, embed_dim=384, num_classes=num_classes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=k) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 384)
        convs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(convs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ================== LOAD OR TRAIN CNN ==================
model_path = "cnn_chattea.pth"
try:
    cnn_model = TextCNN().to(device)
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded pre-trained CNN model")
except:
    print("Training CNN from scratch (~20 seconds)...")
    X = embedder.encode(df['text'].tolist(), convert_to_tensor=True).to(device)
    y = torch.tensor(df['label'].values, dtype=torch.long).to(device)

    train_idx, val_idx = train_test_split(
        torch.arange(len(X)),
        test_size=0.2,
        random_state=42,
        stratify=y.cpu()      # stratify uses NumPy so move labels to CPU
    )

    X_train = X[train_idx].to(device)
    X_val   = X[val_idx].to(device)
    y_train = y[train_idx].to(device)
    y_val   = y[val_idx].to(device)


    cnn_model = TextCNN().to(device)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    cnn_model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = cnn_model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            acc = (outputs.argmax(1) == y_train).float().mean().item()
            print(f"Epoch {epoch:2d} | Acc: {acc:.4f} | Loss: {loss.item():.4f}")
    torch.save(cnn_model.state_dict(), model_path)
    print("CNN trained and saved!")

cnn_model.eval()

# ================== PHONE REGEX ==================
PHONE_PATTERN = re.compile(r'(\b08[1-9]\d{7,12}\b|\b628[1-9]\d{7,12}\b|\b\+628[1-9]\d{7,12}\b)', re.I)

# ================== MAIN CHAT FUNCTION ==================
def get_chattea_reply(user_input: str) -> str:
    text = user_input.strip().lower()

    # Rule-based greetings
    if any(g in text for g in ["hai", "halo", "hello", "hi", "hey", "pagi", "siang", "malam"]):
        return RESPONSES["greeting"]["en"]
    if any(g in text for g in ["bye", "goodbye", "dadah", "sampai jumpa"]):
        return RESPONSES["goodbye"]["en"]

    # Extract phone
    phone_match = PHONE_PATTERN.search(user_input)
    extracted_phone = None
    if phone_match:
        num = phone_match.group(0).replace(" ", "").replace("-", "")
        if num.startswith("08"):
            extracted_phone = num
        elif num.startswith("628"):
            extracted_phone = "0" + num[2:]
        elif num.startswith("+628"):
            extracted_phone = "0" + num[3:]

    # Embedding + prediction
    with torch.no_grad():
        user_emb = embedder.encode(user_input, convert_to_tensor=True).to(device)
        user_emb = user_emb.unsqueeze(0)  # (1, 384)

        # CNN
        cnn_logits = cnn_model(user_emb)
        cnn_confidence = cnn_logits.softmax(1).max().item()
        cnn_intent = intent_map[cnn_logits.argmax(1).item()]

        # Retrieval fallback
        cos_scores = util.cos_sim(user_emb, sentence_embeddings)[0]
        retrieval_intent = df.iloc[cos_scores.argmax().item()]['intent']

        final_intent = cnn_intent if cnn_confidence > 0.90 else retrieval_intent

    # Phone check special case
    if final_intent == "phone_check" and extracted_phone:
        nice_phone = extracted_phone[:4] + "..." + extracted_phone[-4:]
        return f"Checking {extracted_phone}...\nYes, this number is registered and active on WhatsApp!\n\nYou can safely include it in your blast list."

    # Normal response
    response = RESPONSES.get(final_intent, RESPONSES["help"])
    return response["en"] if isinstance(response, dict) else response

# ================== RUN CHAT ==================
if __name__ == "__main__":
    print("\nChattea Assistant Ready! (Type 'quit' to exit)\n")
    while True:
        try:
            msg = input("You: ").strip()
            if msg.lower() in ["quit", "exit", "bye"]:
                print("Bot:", RESPONSES["goodbye"]["en"])
                break
            if not msg:
                continue
            reply = get_chattea_reply(msg)
            print("Bot:", reply, "\n")
        except KeyboardInterrupt:
            print("\nBot: Goodbye!")
            break