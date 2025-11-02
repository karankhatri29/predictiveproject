import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import torch
from collections import Counter
import os

nltk.download('punkt')

class Vocab:
    def __init__(self, min_freq=1):
        self.word2idx = {}
        self.idx2word = []
        self.min_freq = min_freq
        self.counter = Counter()
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
    
    def build_vocab(self, texts):
        for text in texts:
            words = word_tokenize(text.lower())
            self.counter.update(words)
        # Add special tokens first
        self.idx2word = [self.pad_token, self.unk_token]
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        
        for word, freq in self.counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
    
    def word_to_index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_token])
    
    def index_to_word(self, idx):
        if idx < len(self.idx2word):
            return self.idx2word[idx]
        return self.unk_token
    
    def encode_sentence(self, sentence, max_len):
        words = word_tokenize(sentence.lower())
        ids = [self.word_to_index(w) for w in words]
        if len(ids) < max_len:
            ids += [self.word2idx[self.pad_token]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids
    
    def decode_sentence(self, ids):
        return " ".join([self.index_to_word(i) for i in ids])

def load_data(file_path, max_sentences=1000):
    """
    Load raw text file, split into paragraphs -> sentences.
    Return list of paragraphs, each paragraph is list of sentences.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    data = []
    for p in paragraphs[:max_sentences]:
        sents = sent_tokenize(p)
        if len(sents) < 2:
            continue
        data.append(sents)
    return data
