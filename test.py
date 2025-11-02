import pickle

with open("ngram_model.pkl", "rb") as f:
    model = pickle.load(f)

print(f"Number of prefixes: {len(model)}")
print("Example prefix -> next_words:", list(model.items())[:3])
