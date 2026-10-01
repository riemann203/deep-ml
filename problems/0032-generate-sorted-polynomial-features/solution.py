import numpy as np
import math
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.asarray(X, dtype=float)
    output = []
    for row in X:
        features = []
        for d in range(degree+1):
            items = combinations_with_replacement(row, d)
            features.extend(
                math.prod(item) for item in items
            )
        features = sorted(features)
        output.append(features)
    return np.array(output)