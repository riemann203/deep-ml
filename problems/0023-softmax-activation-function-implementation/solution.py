import math

def softmax(scores: list[float]) -> list[float]:
    max_val = max(scores)
    scores = [score - max_val for score in scores]
    exp_scores = [math.exp(score) for score in scores]
    total = sum(exp_scores)
    return [exp_score / total for exp_score in exp_scores]