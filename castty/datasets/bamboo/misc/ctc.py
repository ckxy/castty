import os
import torch
from ..builder import INTERNODE
from ..base_internode import BaseInternode


__all__ = ['CTCEncode']


@INTERNODE.register_module()
class CTCEncode(BaseInternode):
	def __init__(self, char_path, blank_first=False, **kwargs):
		assert os.path.exists(char_path)

		self.char_path = char_path
		self.blank_first = blank_first

		with open(char_path, 'r') as f:
			dict_character = ''.join(f.readlines())
		dict_character = list(dict_character)

		self.mapping = dict()
		for i, char in enumerate(dict_character):
			if self.blank_first:
			# NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
				self.mapping[char] = i + 1
			else:
				self.mapping[char] = i

		if self.blank_first:
			self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)
		else:
			self.character = dict_character + ['[CTCblank]']

		BaseInternode.__init__(self, **kwargs)

	def forward(self, data_dict, **kwargs):
		data_dict['seq_length'] = torch.IntTensor(1).fill_(len(data_dict['seq']))

		seq = list(data_dict['seq'])
		seq = [self.mapping[char] for char in seq]
		data_dict['encoded_seq'] = torch.IntTensor(seq)

		return data_dict

	def __repr__(self):
		return 'CTCEncode(char_path={}, blank_first={})'.format(self.char_path, self.blank_first)
