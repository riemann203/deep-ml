from collections import Counter
import math


def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	counts = Counter(apples)
	frequencies = [count / len(apples) for count in counts.values()]
	return -sum(freq * math.log(freq) for freq in frequencies)
