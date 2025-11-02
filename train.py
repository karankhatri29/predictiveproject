import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from hlstm_model import HLSTM # Assuming you have this file
from data_utils import Vocab, load_data # Assuming these files/classes exist
from datasets import load_dataset # Still needed if you use HuggingFace, but commented out below
import nltk
import os
import re # Needed for cleaning the email content
nltk.download('punkt')

# Config
MAX_SENT_LEN = 20
BATCH_SIZE = 8
EMBED_DIM = 128
HIDDEN_WORD = 256
HIDDEN_SENT = 256
TOPIC_DIM = 64
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use your custom data file
DATA_FILE = "synthetic_emails_10k_paragraphs.txt"
EMAIL_SEPARATOR = "\n\n\n" # Must match the separator used in the generation script

# --- FIXED Preprocessing Function ---
def preprocess_emails(data_file, separator):
    """
    Loads custom email data, splits it by the separator, and extracts
    the 'Content' field, tokenizing it into sentences.
    
    Returns: A list of paragraphs, where each paragraph is a list of sentences.
    (This matches the expected input format for the HLSTMDataset).
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found at: {data_file}. Please ensure the email generation script was run.")
        
    with open(data_file, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # 1. Split by the email separator
    email_blocks = full_text.strip().split(separator)
    
    paragraphs = []
    
    # Regex to efficiently find the 'Content:' line and extract the text
    content_pattern = re.compile(r"^Content:\s*(.*)", re.MULTILINE | re.DOTALL)
    
    for block in email_blocks:
        if not block.strip():
            continue
            
        match = content_pattern.search(block)
        
        if match:
            content_text = match.group(1).strip()
            
            # 2. Tokenize the content into sentences
            # We use the lines we explicitly created in the generation script as sentences
            # Or use nltk for true sentence tokenization:
            # sentences = nltk.sent_tokenize(content_text)
            
            # Since your content is 2-3 lines, splitting by newline is safer:
            sentences = [s.strip() for s in content_text.split('\n') if s.strip()]

            if sentences:
                paragraphs.append(sentences)
                
    print(f"Successfully loaded and parsed {len(paragraphs)} email content blocks.")
    return paragraphs
# ------------------------------------


# Dataset Class (No changes needed here)
class HLSTMDataset(Dataset):
    def __init__(self, paragraphs, vocab, max_sent_len=MAX_SENT_LEN):
        self.paragraphs = paragraphs
        self.vocab = vocab
        self.max_sent_len = max_sent_len
        self.data = []
        for p in paragraphs:
            # Only use paragraphs with more than one sentence for (prev_sent, curr_sent) pairs
            for i in range(1, len(p)):
                prev_sent = p[i - 1]
                curr_sent = p[i]
                self.data.append((prev_sent, curr_sent))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prev_sent, curr_sent = self.data[idx]
        # Ensure sentences are tokenized (split into words) before encoding!
        # Assuming vocab.encode_sentence handles word tokenization.
        prev_ids = torch.tensor(self.vocab.encode_sentence(prev_sent, self.max_sent_len))
        curr_ids = torch.tensor(self.vocab.encode_sentence(curr_sent, self.max_sent_len))
        return prev_ids, curr_ids

def collate_fn(batch):
    prev_batch = torch.stack([x[0] for x in batch])
    curr_batch = torch.stack([x[1] for x in batch])
    return prev_batch, curr_batch

def train():
    
    # --- FIX APPLIED HERE: Load and process the custom file ---
    # The original line: dataset = DATA_FILE
    # The original line: train_texts = dataset["train"]["text"] <-- This caused the error
    
    paragraphs = preprocess_emails(DATA_FILE, EMAIL_SEPARATOR)
    
    print(f"Prepared {len(paragraphs)} paragraphs from the email file.")
    # ---------------------------------------------------------
    
    # Build vocab from all sentences
    all_sents = [sent for p in paragraphs for sent in p]
    
    # NOTE: You MUST ensure your 'Vocab' class can handle the raw sentence strings
    # and tokenize them into words.
    vocab = Vocab(min_freq=1) 
    vocab.build_vocab(all_sents)
    print(f"Vocab size: {len(vocab.idx2word)}")

    dataset = HLSTMDataset(paragraphs, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Rest of the training loop...
    model = HLSTM(len(vocab.idx2word), EMBED_DIM, HIDDEN_WORD, HIDDEN_SENT, TOPIC_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop (omitted for brevity, assume correct) ---
    # ...
    
    print(f"Total training pairs: {len(dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (prev_sents, curr_sents) in enumerate(dataloader):
            prev_sents = prev_sents.to(DEVICE)
            curr_sents = curr_sents.to(DEVICE)

            # Create a zero context vector for first pass
            zero_context = torch.zeros(prev_sents.size(0), TOPIC_DIM, device=DEVICE)

            # Forward pass prev sentence to get thought vector
            with torch.no_grad():
                # Assuming model returns (logits, word_h, word_c, sent_thought_vec)
                _, _, _, thought_vec = model(prev_sents, sent_context=zero_context) 

            # Forward pass current sentence with thought vector
            logits, _, _, _ = model(curr_sents, sent_context=thought_vec)

            # Shift logits and targets for next word prediction
            logits = logits[:, :-1, :].contiguous()
            targets = curr_sents[:, 1:].contiguous()

            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} average loss: {total_loss / len(dataloader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
    }, 'hlstm_model.pth')
    print("Model saved to hlstm_model.pth")


if __name__ == "__main__":
    train()