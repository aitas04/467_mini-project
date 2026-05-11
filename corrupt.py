import pandas as pd
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import spacy
import gensim.downloader as api


nlp = spacy.load("en_core_web_sm")
en_model = api.load("word2vec-google-news-300")

torch.manual_seed(42)
df = pd.read_csv("dataset.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
sent1 = df["sentence1"].astype(str).tolist()
sent2 = df["sentence2"].astype(str).tolist()

labels = df["label"].astype(int).tolist()
split = int(0.8 * len(df))
train_s1, test_s1 = sent1[:split], sent1[split:]
train_s2,test_s2 = sent2[:split], sent2[split:]
train_labels,test_labels = labels[:split], labels[split:]


label_1_s1 = []
label_2_s2 = []
for s1, s2, label in zip(test_s1, test_s2, test_labels):
    if label == 1:
        label_1_s1.append(s1)
        label_2_s2.append(s2)


model_name = "roberta_model"
device = "cpu"
print(f"Using device: {device}")



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
model.eval()

test_enc = tokenizer(test_s1, test_s2,truncation=True,padding=True,max_length=128,return_tensors="pt").to(device)
test_labels_tensor = list(test_labels)
with torch.no_grad():
    test_preds = model(**test_enc).logits.argmax(dim=-1).cpu().tolist()
test_acc = round(accuracy_score(test_labels_tensor, test_preds) * 100, 2)
print(f"Test set accuracy: {test_acc}%")
enc = tokenizer(label_1_s1,label_2_s2,truncation=True,padding=True,max_length=128,return_tensors="pt").to(device)
with torch.no_grad():
    preds = model(**enc).logits.argmax(dim=-1).cpu().tolist()

acc_label_1_s1 = []
acc_label_2_s2 = []

for s1, s2, label in zip(label_1_s1, label_2_s2, preds):
    if label == 1:
        acc_label_1_s1.append(s1)
        acc_label_2_s2.append(s2)



def similar(word, pos, topn=200):
    if word.lower() not in en_model.key_to_index:
        return []
    original_lemma = nlp(word.lower())[0].lemma_
    target = 0.7 if pos == "NOUN" else 0.3
    tolerance = 0.1

    candidates = en_model.most_similar(word.lower(), topn=int(topn))
    docs = nlp.pipe([w for w, _ in candidates], batch_size=16)
    filtered = []
    for doc, (w, score) in zip(docs, candidates):
        if len(doc) == 0:
            continue
        token = doc[0]
        if (token.pos_ == pos and w.lower() != word.lower() and token.lemma_ != original_lemma and "_" not in w and w.isalpha() and w == w.lower() and abs(score - target) <= tolerance):
            filtered.append((w, score))
    return filtered

def replace(word, pos):
    candidates = similar(word, pos)
    if not candidates:
        return word
    return min(candidates, key=lambda x: abs(x[1] - 0.7))[0]




def corrupt(s2):
    doc = nlp(s2)
    new_tokens = [token.text for token in doc]
    for token in doc:
         if token.pos_ in ["NOUN", "VERB"]:
            candidates = similar(token.text, token.pos_)
            if candidates:
                worst = min(candidates, key=lambda x: x[1])
                new_tokens[token.i] = worst[0]
                break
    return " ".join(new_tokens)


corrupt_s1 = []
corrupt_s2 = []

for s1, s2 in zip(acc_label_1_s1, acc_label_2_s2):
    c2 = corrupt(s2)
    corrupt_s1.append(s1)
    corrupt_s2.append(c2)


corrupt_enc = tokenizer(corrupt_s1,corrupt_s2,truncation=True,padding=True,max_length=128,return_tensors="pt").to(device)
with torch.no_grad():
    corrupt_preds = model(**corrupt_enc).logits.argmax(dim=-1).cpu().tolist()

correct_0 = sum(1 for p in corrupt_preds if p == 0)
proportion = round(correct_0 / len(corrupt_preds) * 100, 2)

print(f"correctly labeled: {correct_0}/{len(corrupt_preds)} = {proportion}%")


plt.bar(
    ["Test Set Accuracy", "Corrupted Pairs (labeled 0)"],
    [test_acc, proportion]
)
plt.title("Model Performance")
plt.ylabel("Accuracy %")
plt.ylim(0, 100)
plt.savefig("model_performance.png")
plt.close()

incorrect_1 = len(corrupt_preds) - correct_0

plt.bar(["Correctly classified as 0", "Incorrectly classified as 1"],[correct_0, incorrect_1])
plt.title("Model Performance on Corrupted Paraphrase Pairs")
plt.ylabel("Number of samples")
plt.savefig("corruption_performance.png")
plt.close()

