import math
from collections import Counter, defaultdict


def calculate_entropy(labels: list[str]) -> float:
    """Calculate the entropy of a list of labels."""
    counts = Counter(labels)
    total = len(labels)
    entropy = 0.0
    for count in counts.values():
        freq = count / total
        entropy -= freq * math.log(freq)
    return entropy

def calculate_information_gain(examples: list[dict], attr: str, target_attr: str) -> float:
    """Calculate the information gain of splitting on attr."""
    labels = [ex[target_attr] for ex in examples]
    entropy = calculate_entropy(labels)

    total = len(examples)
    conditional_entropy = 0.0

    group_by_attr = defaultdict(list)
    for ex in examples:
        group_by_attr[ex[attr]].append(ex)

    for group in group_by_attr.values():
        group_labels = [ex[target_attr] for ex in group]
        conditional_entropy += len(group) / total * calculate_entropy(group_labels)
    return entropy - conditional_entropy

def majority_class(examples: list[dict], target_attr: str) -> str:
    """Return the majority class. Break ties alphabetically."""
    counts = Counter(ex[target_attr] for ex in examples)
    max_count = max(counts.values())
    return min(
        label
        for label, count in counts.items()
        if count == max_count
    )

def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
    """Build a decision tree using the ID3 algorithm."""
    labels = [ex[target_attr] for ex in examples]
    if len(set(labels)) == 1:
        return labels[0]
    if not attributes:
        return majority_class(examples, target_attr)

    best_info_gain = -float("inf")
    best_attr = None
    for attr in attributes:
        info_gain = calculate_information_gain(examples, attr, target_attr)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attr = attr

    groups = defaultdict(list)
    for ex in examples:
        groups[ex[best_attr]].append(ex)

    remaining_attributes = [
        attr for attr in attributes
        if attr != best_attr
    ]

    tree = {best_attr: {}}
    for value in sorted(groups):
        tree[best_attr][value] = learn_decision_tree(
            groups[value],
            remaining_attributes,
            target_attr
        )

    return tree