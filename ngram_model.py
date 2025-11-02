import pickle
import nltk
from collections import defaultdict, Counter

nltk.download('punkt', quiet=True)

# ---------------- CONFIG ----------------
NGRAM_N = 3  # Trigram
EMAIL_FILE = "synthetic_emails_10k_paragraphs.txt"
MODEL_FILE = "ngram_model.pkl"
# ---------------------------------------

# 1️⃣ Load emails
with open(EMAIL_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Split emails by double newlines or separator lines (adjust if needed)
emails = [b.strip() for b in text.split("\n\n") if b.strip()]

# 2️⃣ Extract body text
bodies = []
for email in emails:
    lines = email.splitlines()
    body_started = False
    body_lines = []

    for line in lines:
        if body_started:
            body_lines.append(line.strip())
        if line.lower().startswith("content:"):
            # Start capturing the body
            body_started = True

    if body_lines:
        bodies.append(" ".join(body_lines))

print(f"Loaded {len(bodies)} email bodies for training.")

# 3️⃣ Build N-gram model
ngram_model = defaultdict(Counter)

for sent in bodies:
    tokens = ['<s>'] + nltk.word_tokenize(sent.lower()) + ['</s>']
    for i in range(len(tokens) - NGRAM_N + 1):
        prefix = tuple(tokens[i:i+NGRAM_N-1])
        next_word = tokens[i+NGRAM_N-1]
        ngram_model[prefix][next_word] += 1

# 4️⃣ Save the N-gram model
with open(MODEL_FILE, "wb") as f:
    pickle.dump(ngram_model, f)

print(f"N-gram model built and saved to {MODEL_FILE}")
print(f"Number of prefixes: {len(ngram_model)}")
print("Sample prefixes:", list(ngram_model.items())[:5])
