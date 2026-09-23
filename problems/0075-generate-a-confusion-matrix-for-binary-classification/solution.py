
from collections import Counter

def confusion_matrix(data):
	data = [tuple(item) for item in data]
	counts = Counter(data)
	return [
		[counts[(1, 1)], counts[(1, 0)]],
		[counts[(0, 1)], counts[(0, 0)]]
	]	
