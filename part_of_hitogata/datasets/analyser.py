import copy
import bisect
import numpy as np
from tqdm import tqdm


def image_analysis(dataset, **kwargs):
	mode = kwargs['mode']

	if mode == 'aspect_ratio':
		info = []
		for i in tqdm(range(len(dataset))):
			info_dict = dataset.get_data_info(i)
			info.append(info_dict['h'] / info_dict['w'])
		return quantize(info, kwargs['split'])
	elif mode == 'len':
		return len(dataset)
	else:
		raise ValueError


def quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

