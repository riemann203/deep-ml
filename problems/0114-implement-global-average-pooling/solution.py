import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# 對每一個 channel，把 (height, width) 平均掉
	return np.mean(x, axis=(0, 1))