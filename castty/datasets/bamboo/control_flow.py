import random
from .builder import INTERNODE
from .builder import build_internode
from .base_internode import BaseInternode


__all__ = ['ChooseOne', 'ChooseSome', 'ChooseABranchByID', 'RandomWarpper', 'ForwardOnly', 'BackwardOnly']


@INTERNODE.register_module()
class ChooseOne(BaseInternode):
	def __init__(self, branchs, **kwargs):
		assert len(branchs) > 1

		self.branchs = []
		for branch in branchs:
			self.branchs.append(build_internode(branch, **kwargs))

	def calc_intl_param_forward(self, data_dict):
		intl_bid = random.randint(0, len(self.branchs) - 1)
		return dict(intl_bid=intl_bid)

	def forward(self, data_dict, intl_bid, **kwargs):
		i = self.branchs[intl_bid]
		return i(data_dict)

	def __repr__(self):
		split_str = [i.__repr__() for i in self.branchs]
		bamboo_str = ''
		for i in range(len(split_str)):
			bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
		bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

		return 'ChooseOne(\n  internodes:{}\n )'.format(bamboo_str)


@INTERNODE.register_module()
class ChooseSome(ChooseOne):
	def __init__(self, branchs, num=2, **kwargs):
		assert num > 1 and num <= len(branchs)
		self.num = num
		super(ChooseSome, self).__init__(branchs, **kwargs)

	def calc_intl_param_forward(self, data_dict):
		intl_bids = random.sample(list(range(len(self.branchs))), self.num)
		return dict(intl_bids=intl_bids)

	def forward(self, data_dict, intl_bids, **kwargs):
		for i in intl_bids:
			data_dict = self.branchs[i](data_dict)
		return data_dict

	def __repr__(self):
		split_str = [i.__repr__() for i in self.branchs]
		bamboo_str = ''
		for i in range(len(split_str)):
			bamboo_str += '\n  ' + split_str[i].replace('\n', '\n  ')
		bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

		return 'ChooseSome(\n num={}\n internodes:{}\n )'.format(self.num, bamboo_str)


@INTERNODE.register_module()
class ChooseABranchByID(ChooseOne):
	def __init__(self, branchs, tag='branch_id', **kwargs):
		super(ChooseABranchByID, self).__init__(branchs, **kwargs)
		self.tag = 'intl_' + tag

	def forward(self, data_dict, **kwargs):
		data_dict = self.branchs[data_dict[self.tag]](data_dict)
		return data_dict

	def __repr__(self):
		split_str = [i.__repr__() for i in self.branchs]
		bamboo_str = ''
		for i in range(len(split_str)):
			bamboo_str += f'\n  {i}:' + split_str[i].replace('\n', '\n  ')
		bamboo_str = '(\n{}\n  )'.format(bamboo_str[1:])

		return 'ChooseABranchByID(\n internodes:{}\n )'.format(bamboo_str)


class InternodeWarpper(BaseInternode):
	def __init__(self, internode, **kwargs):
		self.internode = build_internode(internode, **kwargs)


@INTERNODE.register_module()
class RandomWarpper(InternodeWarpper):
	def __init__(self, internode, p=0.5, **kwargs):
		assert 0 < p < 1
		internode.pop('p')

		self.p = p
		super(RandomWarpper, self).__init__(internode, **kwargs)

	def calc_intl_param_forward(self, data_dict):
		intl_random_flag = random.random() < self.p
		return dict(intl_random_flag=intl_random_flag)

	def forward(self, data_dict, intl_random_flag, **kwargs):
		if intl_random_flag:
			data_dict = self.internode(data_dict)
		return data_dict

	def __repr__(self):
		return 'Random_p{:.2f}_'.format(self.p) + self.internode.__repr__()

	def rper(self):
		return 'Random_' + type(self.internode).__name__ + '(not available)'


@INTERNODE.register_module()
class ForwardOnly(InternodeWarpper):
	def __init__(self, internode, **kwargs):
		internode.pop('one_way')
		super(ForwardOnly, self).__init__(internode, **kwargs)

	def forward(self, data_dict):
		return self.internode(data_dict)

	def __repr__(self):
		return 'ForwardOnly_' + self.internode.__repr__()

	def rper(self):
		return 'ForwardOnly_' + type(self.internode).__name__ + '(not available)'


@INTERNODE.register_module()
class BackwardOnly(InternodeWarpper):
	def __init__(self, internode, **kwargs):
		internode.pop('one_way')
		super(BackwardOnly, self).__init__(internode, **kwargs)

	def backward(self, kwargs):
		return self.internode.reverse(**kwargs)
		
	def __repr__(self):
		return 'BackwardOnly_' + type(self.internode).__name__ + '(not available)'

	def rper(self):
		return 'BackwardOnly_' + self.internode.__repr__()
