from math import log2
from collections import Counter, defaultdict
import copy

# ---------- Dataset (Play Tennis) ----------
data = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "Play": "No"},
]

features = ["Outlook", "Temperature", "Humidity", "Wind"]
target = "Play"

# ---------- Helper functions ----------
def entropy(rows, target):
    counts = Counter(r[target] for r in rows)
    total = len(rows)
    return -sum((c / total) * log2(c / total) for c in counts.values())

def partition(rows, feature):
    groups = defaultdict(list)
    for r in rows:
        groups[r[feature]].append(r)
    return groups

def info_gain(rows, feature, target):
    base = entropy(rows, target)
    groups = partition(rows, feature)
    total = len(rows)
    rem = sum((len(g) / total) * entropy(g, target) for g in groups.values())
    return base - rem

def majority_class(rows, target):
    counts = Counter(r[target] for r in rows)
    return counts.most_common(1)[0][0]

# ---------- ID3 algorithm ----------
def id3(rows, features, target):
    labels = [r[target] for r in rows]
    if len(set(labels)) == 1:
        return {"leaf": labels[0]}
    if not features:
        return {"leaf": majority_class(rows, target)}

    gains = [(f, info_gain(rows, f, target)) for f in features]
    best_feature, _ = max(gains, key=lambda x: x[1])

    tree = {"feature": best_feature, "children": {}}
    groups = partition(rows, best_feature)
    remaining = [f for f in features if f != best_feature]
    for val, subset in groups.items():
        if not subset:
            tree["children"][val] = {"leaf": majority_class(rows, target)}
        else:
            tree["children"][val] = id3(subset, remaining, target)

    tree["fallback"] = majority_class(rows, target)
    return tree

# ---------- Prediction ----------
def predict(tree, sample):
    while "leaf" not in tree:
        feat = tree["feature"]
        val = sample.get(feat, None)
        child = tree["children"].get(val)
        if child is None:
            return tree.get("fallback")
        tree = child
    return tree["leaf"]

# ---------- Pretty printing ----------
def print_tree(tree, indent=""):
    if "leaf" in tree:
        print(indent + "→ " + tree["leaf"])
        return
    feat = tree["feature"]
    print(indent + f"[{feat}]")
    for val, child in tree["children"].items():
        print(indent + f"  ├─ {val}:")
        print_tree(child, indent + "  │   ")
    print(indent + f"  (fallback → {tree.get('fallback')})")

# ---------- Build and test ----------
tree = id3(data, features, target)
print("=== Learned ID3 Decision Tree ===")
print_tree(tree)

new_sample = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Wind": "Strong"}
print("\nNew sample:", new_sample)
print("Predicted class:", predict(copy.deepcopy(tree), new_sample))
