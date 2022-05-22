from .base_internode import BaseInternode
from .builder import build_internode
from .builder import INTERNODE
import random


@INTERNODE.register_module()
class ChooseOne(BaseInternode):
	def __init__(self, branchs, **kwargs):
		assert len(branchs) > 1

		self.branchs = []

		for branch in branchs:
			self.branchs.append(build_internode(branch))

	def __call__(self, data_dict):
		i = random.choice(self.branchs)
		return i(data_dict)

	def __repr__(self):
		split_str = [i.__repr__() for i in self.branchs]
		bamboo_str = ''
		for i in range(len(split_str)):
			bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
		bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

		return 'ChooseOne(\n  internodes:{} \n)'.format(bamboo_str)


@INTERNODE.register_module()
class ChooseSome(ChooseOne):
	def __init__(self, branchs, num=2, **kwargs):
		assert num > 1 and num <= len(branchs)
		self.num = num
		super(ChooseSome, self).__init__(branchs, **kwargs)

	def __call__(self, data_dict):
		branchs = random.sample(self.branchs, self.num)
		for branch in branchs:
			data_dict = branch(data_dict)
		return data_dict

	def __repr__(self):
		split_str = [i.__repr__() for i in self.branchs]
		bamboo_str = ''
		for i in range(len(split_str)):
			bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
		bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

		return 'ChooseSome(\n num={}\n internodes:{} \n)'.format(self.num, bamboo_str)


@INTERNODE.register_module()
class RandomWarpper(BaseInternode):
	def __init__(self, internode, p=0.5, **kwargs):
		assert 0 < p < 1
		internode.pop('p')

		self.p = p
		self.internode = build_internode(internode)

	def __call__(self, data_dict):
		if random.random() < self.p:
			data_dict = self.internode(data_dict)
		return data_dict

	def __repr__(self):
		return 'Random_p{:.2f}_'.format(self.p) + self.internode.__repr__()

	def rper(self):
		return 'Random_' + type(self.internode).__name__ + '(not available)'


@INTERNODE.register_module()
class ForwardOnly(BaseInternode):
	def __init__(self, internode, **kwargs):
		internode.pop('one_way')
		self.internode = build_internode(internode)

	def __call__(self, data_dict):
		return self.internode(data_dict)

	def __repr__(self):
		return 'ForwardOnly_' + self.internode.__repr__()

	def rper(self):
		return 'ForwardOnly_' + type(self.internode).__name__ + '(not available)'


@INTERNODE.register_module()
class BackwardOnly(BaseInternode):
	def __init__(self, internode, **kwargs):
		internode.pop('one_way')
		self.internode = build_internode(internode)

	def reverse(self, **kwargs):
		return self.internode.reverse(**kwargs)
		
	def __repr__(self):
		return 'BackwardOnly_' + type(self.internode).__name__ + '(not available)'

	def rper(self):
		return 'BackwardOnly_' + self.internode.__repr__()
