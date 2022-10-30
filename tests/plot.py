import torch
import torch.nn as nn
import math
import numpy as np


def crop_area(polys):
	w, h = 10, 10
	h_array = np.zeros(h, dtype=np.int32)
	w_array = np.zeros(w, dtype=np.int32)
	for points in polys:
		print(points)
		min_x = np.min(points[:, 0])
		max_x = np.max(points[:, 0])
		print(min_x, max_x, max_x - min_x)
		# points = np.round(
		# 	points, decimals=0).astype(np.int32).reshape(-1, 2)
		points = np.ceil(points).astype(np.int32).reshape(-1, 2)
		print(points)
		min_x = np.min(points[:, 0])
		max_x = np.max(points[:, 0])
		print(min_x, max_x, max_x - min_x)
		w_array[min_x:max_x] = 1
		print(np.sum(w_array))
		min_y = np.min(points[:, 1])
		max_y = np.max(points[:, 1])
		h_array[min_y:max_y] = 1
	print(w_array)
	print(h_array)

	# ensure the cropped area not across a text
	h_axis = np.where(h_array == 0)[0]
	w_axis = np.where(w_array == 0)[0]

	print(w_axis)
	print(h_axis)


if __name__ == "__main__":
	points = [np.array([[0.9, 1], [3.4, 2]])]
	crop_area(points)
	# x = np.arange(16)
	# x = torch.from_numpy(x)
	# x = x.view(4, 4)

	# print(x, x.shape)
	# x = x.unsqueeze(-1)

	# x0 = x[0::2, 0::2, :]  # B H/2 W/2 C
	# x1 = x[1::2, 0::2, :]  # B H/2 W/2 C
	# x2 = x[0::2, 1::2, :]  # B H/2 W/2 C
	# x3 = x[1::2, 1::2, :]  # B H/2 W/2 C

	# print(x0[..., 0])
	# print(x1[..., 0])
	# print(x2[..., 0])
	# print(x3[..., 0])

	# x = torch.cat([x0, x1, x2, x3], -1)

	# print(x, x.shape)
	# print(x.flatten())
	# print(x.reshape(-1))
