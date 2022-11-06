import cv2
import math
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_polygon(img, polygons, ignore_flags=None, class_ids=None, classes=None):
	if ignore_flags is None:
		ignore_flags = [False] * len(polygons)

	if class_ids is None:
		class_ids = [0] * len(polygons)

	num_classes = len(classes) if classes else 1
    
	hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
	# print(colors, 'ccc')

	if not isinstance(img, Image.Image):
		is_np = True
		img = Image.fromarray(img)
	else:
		is_np = False

	w, h = img.size
	l = math.sqrt(h * h + w * w)
	draw = ImageDraw.Draw(img)

	for polygon, ignore_flag, cid in zip(polygons, ignore_flags, class_ids):
		if ignore_flag:
			draw.polygon(polygon.astype(np.int32).flatten().tolist(), outline=(40, 40, 40), width=2)
		else:
			draw.polygon(polygon.astype(np.int32).flatten().tolist(), outline=colors[cid], width=2)

	if is_np:
		img = np.asarray(img)

	return img


def get_cw_order_form(polygon):
	# polygon = np.array([[20, 0], [0, 0], [0, 20], [20, 20]])
	# print('----in----')
	# print(polygon, polygon.shape)

	n = len(polygon)
	center = np.mean(polygon, axis=0)
	distances = np.power(polygon[..., 0], 2) + np.power(polygon[..., 1], 2)
	p1_id = np.argmin(distances)
	p2_id = (p1_id + 1) % n

	x1, y1 = polygon[p1_id] - center
	x2, y2 = polygon[p2_id] - center
	cross_p1p2 = x1 * y2 - x2 * y1

	if cross_p1p2 > 0:
		res = np.roll(polygon, n - p1_id, 0)
	elif cross_p1p2 < 0:
		# print('----in----')
		res = np.roll(polygon, n - p1_id, 0)
		# print(res)
		res = res[::-1, ...]
		# print(res)
		res = np.roll(res, 1, 0)
		# print(res)
		# print('----out---')
	else:
		raise ValueError

	return res
