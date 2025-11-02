from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset["train"]["text"]
#print(train_texts[:5])