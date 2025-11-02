import torch
import torch.nn as nn

class HLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim_word, hidden_dim_sent, topic_dim=None):
        super(HLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.topic_dim = topic_dim if topic_dim is not None else 0
        input_size = embedding_dim + self.topic_dim
        print(f"Initializing word_lstm with input_size={input_size}")
        
        self.word_lstm = nn.LSTM(input_size, hidden_dim_word, batch_first=True)
        self.sent_lstm = nn.LSTM(hidden_dim_word, hidden_dim_sent, batch_first=True)
        
        if topic_dim:
            self.topic_proj = nn.Linear(hidden_dim_sent, topic_dim)
        else:
            self.topic_proj = None
        
        self.fc_out = nn.Linear(hidden_dim_word, vocab_size)
    
    def forward(self, word_inputs, sent_context=None, hidden_word=None, hidden_sent=None):
        embed = self.embedding(word_inputs)  # (batch, seq_len, embedding_dim)
        
        if sent_context is not None:
            if sent_context.size(1) != self.topic_dim:
                raise ValueError(f"sent_context dimension {sent_context.size(1)} does not match topic_dim {self.topic_dim}")
            sent_context_exp = sent_context.unsqueeze(1).expand(-1, embed.size(1), -1)
            lstm_input = torch.cat([embed, sent_context_exp], dim=2)  # (batch, seq_len, embedding_dim + topic_dim)
        else:
            # If topic_dim > 0, we must pass sent_context, else error will happen
            if self.topic_dim > 0:
                raise RuntimeError("sent_context must be provided when topic_dim > 0")
            lstm_input = embed
        
        # Debug print
        # print(f"lstm_input.shape = {lstm_input.shape}")
        
        word_out, hidden_word = self.word_lstm(lstm_input, hidden_word)  # (batch, seq_len, hidden_dim_word)
        
        sent_embed = word_out[:, -1, :]  # (batch, hidden_dim_word)
        sent_embed_unsq = sent_embed.unsqueeze(1)  # (batch, 1, hidden_dim_word)
        
        sent_out, hidden_sent = self.sent_lstm(sent_embed_unsq, hidden_sent)  # (batch, 1, hidden_dim_sent)
        sent_out = sent_out.squeeze(1)  # (batch, hidden_dim_sent)
        
        if self.topic_proj:
            topic_vec = self.topic_proj(sent_out)  # (batch, topic_dim)
        else:
            topic_vec = sent_out
        
        logits = self.fc_out(word_out)  # (batch, seq_len, vocab_size)
        
        return logits, hidden_word, hidden_sent, topic_vec
