import math
from collections import Counter
import numpy as np

def calculate_entropy(labels: list) -> float:
    """Calculate the entropy of a list of labels."""
    labels = np.array(labels)
    _, counts = np.unique(labels, return_counts=True)
    probs = counts/sum(counts)
    return -np.sum([prob*np.log2(prob) if prob!=0 else 0 for prob in probs])

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    parent_entropy = calculate_entropy(example[target_attr] for example in examples)

    values = [example[attr] for example in examples]
    unique_vals = np.unique(values)

    weighted_child_entropy = 0.0
    for val in unique_vals:
        subset = [ex for ex in examples if ex[attr] == val]
        labels = [ex[target_attr] for ex in subset]  
        weight = len(subset) / len(examples)
        #print(val, subset, labels, weight)
        weighted_child_entropy += weight * calculate_entropy(labels)

    return parent_entropy - weighted_child_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    labels = [ex[target_attr] for ex in examples]
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [cls for cls, cnt in counts.items() if cnt == max_count]
    return sorted(candidates)[0]

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    # Base case 1: all examples have same label
    labels = [ex[target_attr] for ex in examples]
    if len(set(labels)) == 1:
        return labels[0]

    # Base case 2: no attributes left
    if not attributes:
        return majority_class(examples, target_attr)

    # Choose best attribute by information gain
    gains = [(attr, calculate_information_gain(examples, attr, target_attr)) for attr in attributes]
    best_attr, _ = max(gains, key=lambda x: x[1])

    # Build tree
    tree = {best_attr: {}}
    values = np.unique([ex[best_attr] for ex in examples])

    for val in values:
        subset = [ex for ex in examples if ex[best_attr] == val]
        if not subset:
            tree[best_attr][val] = majority_class(examples, target_attr)
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = learn_decision_tree(subset, remaining_attrs, target_attr)

    return tree