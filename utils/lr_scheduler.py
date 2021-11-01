import math
import importlib


def create_lr_scheduler(cfg, **kwargs):
    schedulerlib = importlib.import_module('utils.lr_scheduler')

    lr_scheduler = None
    target_name = cfg.name.replace('_', '') + 'scheduler'
    for name, cls in schedulerlib.__dict__.items():
        if name.lower() == target_name.lower() \
           and issubclass(cls, BaseScheduler):
            scheduler = cls

    if scheduler is None:
        raise NotImplementedError(
            "{}.py中，没有以{}的小写为名且是BaseScheduler的衍生类的类".format(name, target_name))

    return scheduler(cfg, **kwargs)


class BaseScheduler(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.use_epoch = cfg.use_epoch
        self.warmup = cfg.warmup if cfg.warmup else 0
        self.wu_is_contant = cfg.is_contant if cfg.is_contant else False
        if not self.use_epoch:
            self.max_n = kwargs['steps_per_epoch'] * kwargs['max_epoches']
            self.wu_n = kwargs['steps_per_epoch'] * cfg.warmup
        else:
            self.max_n = kwargs['max_epoches']
            self.wu_n = cfg.warmup

    def __call__(self, optimizer, epoch, step):
        if self.use_epoch:
        	n = epoch - 1
        else:
        	n = step

        if n <= self.wu_n and self.wu_n > 0:
        	lr = self.get_warm_up_lr(n)
        else:
        	lr = self.get_current_lr(n)

        for param_group in optimizer.param_groups:
        	param_group['lr'] = lr

        return lr

    def get_current_lr(self, n):
    	raise NotImplementedError

    def get_warm_up_lr(self, n):
        if self.wu_is_contant:
            return self.cfg.min_lr
        else:
            return self.cfg.min_lr + (self.cfg.max_lr - self.cfg.min_lr) * n / self.wu_n

    def __repr__(self):
        if self.use_epoch:
        	return 'max_epochs={}, warmup_epochs={}'.format(self.max_n, self.wu_n)
        else:
        	return 'max_steps={}, warmup_steps={}'.format(self.max_n, self.wu_n)


class CosineScheduler(BaseScheduler):
    def __init__(self, cfg, **kwargs):
    	super(CosineScheduler, self).__init__(cfg, **kwargs)

    def get_current_lr(self, n):
    	return self.cfg.min_lr + (self.cfg.max_lr - self.cfg.min_lr) * 0.5 * (1 + math.cos((n - self.wu_n) * math.pi / (self.max_n - self.wu_n)))

    def __repr__(self):
        s = super(CosineScheduler, self).__repr__()
        return 'CosineScheduler(min_lr={}, max_lr={}, {})'.format(self.cfg.min_lr, self.cfg.max_lr, s)


class StepsScheduler(BaseScheduler):
    def __init__(self, cfg, **kwargs):
    	super(StepsScheduler, self).__init__(cfg, **kwargs)
    	assert len(self.cfg.milestones) > 0
    	if len(self.cfg.milestones) > 1:
    		for i in range(1, len(self.cfg.milestones)):
    			if self.cfg.milestones[i] <= self.cfg.milestones[i - 1]:
    				raise ValueError
    	self.idx = 0
    	self.current_lr = self.cfg.max_lr

    def get_current_lr(self, n):
    	if self.idx >= len(self.cfg.milestones):
    		return self.current_lr
    	else:
    		if n >= self.cfg.milestones[self.idx]:
    			self.current_lr *= self.cfg.gamma
    			self.idx += 1
    		return self.current_lr

    def __repr__(self):
        s = super(StepsScheduler, self).__repr__()
        if not self.cfg.min_lr:
        	self.cfg.min_lr = None
        return 'StepsScheduler(gamma={}, milestones={}, min_lr={}, max_lr={}, {})'.format(self.cfg.gamma, self.cfg.milestones, self.cfg.min_lr, self.cfg.max_lr, s)


if __name__ == '__main__':
    from addict import Dict
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    s = Dict(dict(
        name='cosine',
        use_epoch=False,
        warmup=2,
        is_contant=False,
        min_lr=1e-6,
        max_lr=2e-4
    ))

    # lrs = create_lr_scheduler(cfg.train.scheduler, steps_per_epoch=steps_per_epoch, max_epoches=cfg.train.max_epoches)
    lrs = CosineScheduler(s, steps_per_epoch=1380, max_epoches=100)
    print(lrs)

    l = []
    s = 0
    for i in tqdm(range(1, 101)):
        for j in range(1380):
            lr = lrs(None, i, s)
            s += 1
            l.append(lr)

    print(s)
    print(len(l))

    plt.plot(np.array(range(len(l))), np.array(l))
    plt.show()
    np.savetxt('a.txt', np.array(l))