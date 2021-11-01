import math
import colorsys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_polygon_without_label(img, polygons):
    w, h = img.size
    l = math.sqrt(h * h + w * w)
    draw = ImageDraw.Draw(img)

    for polygon in polygons:
        # print(polygon.astype(np.int).flatten().tolist())
        draw.polygon(polygon.astype(np.int).flatten().tolist(), outline=(0, 255, 255))
        # draw.polygon([(0, 0), (100, 20), (20, 50)], outline=(0, 255, 255))
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
