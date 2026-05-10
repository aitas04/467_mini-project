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

acc_label_1_s1 = label_1_s1.copy()
acc_label_2_s2 = label_2_s2.copy()

for s1, s2, label in zip(label_1_s1, label_2_s2, preds):
    if label == 1:
        acc_label_1_s1.append(s1)
        acc_label_2_s2.append(s2)

def get_similar_nouns(word, topn=50, threshold=0.6):
    if word not in en_model.key_to_index:
        return []

    candidates = en_model.most_similar(word, topn=topn)
    words = [w for w, _ in candidates]

    docs = nlp.pipe(words, batch_size=16)

    filtered = []
    for doc, (w, score) in zip(docs, candidates):
        if len(doc) > 0 and doc[0].pos_ == "NOUN" and score >= threshold:
            filtered.append((w, score))

    return filtered


def replace(word, target_score=0.7):
    candidates = get_similar_nouns(word)

    if not candidates:
        return word

    best_word = word
    best_diff = float("inf")

    for w, score in candidates:
        diff = abs(score - target_score)
        if diff < best_diff:
            best_diff = diff
            best_word = w

    return best_word

def corrupt(s2):
    doc = nlp(s2)
    new_tokens = [token.text for token in doc]

    for token in doc:
        if token.pos_ in ["NOUN", "VERB"]:
            replacement = replace(token.text)
            if replacement != token.text:
                new_tokens[token.i] = replacement
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

