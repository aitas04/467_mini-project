import random
import csv

def load_words(file):
    words = []
    with open(file) as f:
        for line in f:
            w = "".join(line.split()).lower()
            if w:
                words.append(w)
    return list(set(words))

nouns = load_words("nouns.txt")
verbs = load_words("tv.txt")
random.shuffle(nouns)
random.shuffle(verbs)
nouns = nouns[:800]
verbs = verbs[:400]

def pasttense(v):
    if v.endswith("ed"):
        return v
    if v.endswith("e"):
        return v + "d"
    elif v.endswith("y") and len(v) > 1 and v[-2] not in "aeiou":
        return v[:-1] + "ied"
    elif (len(v) > 2 and
          v[-1] not in "aeiou" and
          v[-2] in "aeiou" and
          v[-3] not in "aeiou"):
        return v + v[-1] + "ed"
    else:
        return v + "ed"

def paraphrase_versions(n1, n2, v):
    vt = pasttense(v)
    return [
        f"The {n1} {vt} the {n2}",
        f"The {n2} was {vt} by the {n1}",
        f"Who the {n2} was {vt} by was the {n1}",
        f"Who {vt} the {n2} was the {n1}",
        f"It was the {n2} that the {n1} {vt}",
        f"The {n2} was what the {n1} {vt}",
    ]

def reversed_versions(n1, n2, v):
    vt = pasttense(v)
    return [
        f"The {n2} {vt} the {n1}",
        f"The {n1} was {vt} by the {n2}",
        f"Who the {n1} was {vt} by was the {n2}",
        f"Who {vt} the {n1} was the {n2}",
        f"It was the {n1} that the {n2} {vt}",
        f"The {n1} was what the {n2} {vt}",
    ]

def build_dataset():
    data = []
    noun_pairs = []
    for i in range(0, len(nouns), 2):
        noun_pairs.append((nouns[i], nouns[i+1]))
        
    for i in range(200):
        n1, n2 = noun_pairs[i]
        v = verbs[i]
        variants = paraphrase_versions(n1, n2, v)
        s1, s2 = random.sample(variants, 2)
        data.append((1, s1, s2))

    
    for i in range(200):
        n1, n2 = noun_pairs[200 + i]
        v = verbs[200 + i]

       
        s1 = random.choice(paraphrase_versions(n1, n2, v))

        
        s2 = random.choice(reversed_versions(n1, n2, v))

        data.append((0, s1, s2))

    random.shuffle(data)
    return data

data = build_dataset()

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "sentence1", "sentence2"])
    for label, s1, s2 in data:
        writer.writerow([label, s1, s2])
