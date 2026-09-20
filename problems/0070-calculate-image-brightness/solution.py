import numpy as np


def calculate_brightness(img):
	if not img or not img[0]:
		return -1

	width = len(img[0])
	if any(len(row) != width for row in img):
		return -1

	img = np.asarray(img)
	if np.any((img < 0) | (img > 255)):
		return -1

	return round(float(np.mean(img)), 2)
