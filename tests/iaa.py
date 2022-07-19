import os
import cv2
import math
import time
import torch
import ntpath
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from part_of_hitogata.utils.polygon_tools import *
import imgaug as ia
import imgaug.augmenters as iaa

from part_of_hitogata.datasets.bamboo.warp import WarpPerspective


def aug_kp(aug, polys, size):
	res = []
	for poly in polys:
		kp = [ia.Keypoint(p[0], p[1]) for p in poly]
		kp = aug.augment_keypoints([ia.KeypointsOnImage(kp, shape=size)])[0].keypoints
		r = [[p.x, p.y] for p in kp]
		r = np.asarray(r)
		res.append(r)
	return res


if __name__ == '__main__':
	np.set_printoptions(precision=4, suppress=True)
	# img = Image.open('../datasets/ICDAR2015/ch4_training_images/img_1.jpg').convert('RGB')
	# img = cv2.imread('../datasets/ICDAR2015/ch4_training_images/img_1.jpg')
	img = cv2.imread('a.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	polys, flags = torch.load('p.pth')
	# print(flags)
	# print(img, polys)

	# plt.figure()
	# plt.imshow(img)

	p = [[ 39.440895,  23.079351], [505.36182,   87.89065 ], [505.36182,   89.89065 ],[ 18.866693,  85.40677 ]]

	# wp = WarpPerspective(expand=True)
	# iaa_img = wp(dict(image=img))['image']

	# iaa_img = draw_polygon_without_label(iaa_img, [np.asarray(p)])
	# plt.figure()
	# plt.imshow(iaa_img)
	# plt.axis('off')
	# plt.show()
	# exit()

	seq = iaa.Sequential([
		iaa.PerspectiveTransform(scale=(0.15, 0.15), fit_output=True)
	], random_order=True)

	while True:
		iaa_polys = [ia.Polygon(p) for p in polys]

		iaa_img, iaa_polys = seq(image=img, polygons=iaa_polys)
		iaa_polys = [p.exterior for p in iaa_polys]

		iaa_img = draw_polygon_without_label(iaa_img, iaa_polys, flags.get('ignore_flag'))

		plt.figure()
		plt.imshow(iaa_img)
		plt.axis('off')
		plt.show()
		# break

	# while True:
	# 	aug = seq.to_deterministic()
	# 	iaa_img = aug.augment_image(img)
	# 	iaa_polys = aug_kp(aug, polys, img.shape[:2])

	# 	iaa_img = draw_polygon_without_label(iaa_img, iaa_polys)

	# 	plt.imshow(iaa_img)
	# 	plt.axis('off')
	# 	plt.show()
