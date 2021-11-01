import os
import math
import logging
import numpy as np
from PIL import Image
from visualization.progress import decode_format_dict

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


__all__ = ['Visualizer']


class VIS(object):
    def __init__(self, save_path, use_visdom):
        self.save_path = os.path.join(save_path, self.__class__.__name__.lower().replace('vis', ''))
        os.makedirs(self.save_path, exist_ok=True)

        self.use_visdom = use_visdom
        self.data = dict()

    def parse_data(self, k, v, epoch, total_steps):
        raise NotImplementedError

    def visdom_visualize(self, k):
        raise NotImplementedError

    def restore(self, vis):
        raise NotImplementedError

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data

    def matplotlib_visualize(self, k, epoch, total_steps):
        raise NotImplementedError

    def visualize(self, od, epoch, total_steps, vis):
        for k, v in od.items():
            save_plt = v['save'] if 'save' in v.keys() else False
            assert isinstance(save_plt, bool)

            self.parse_data(k, v, epoch, total_steps)
            if self.use_visdom:
                self.data[k]['opts'] = v['opts']
                self.visdom_visualize(k, vis)
            if save_plt:
                self.matplotlib_visualize(k, epoch, total_steps)
        # exit()


class VISLine(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISLine, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        update = not 'x' in v.keys()

        if k not in self.data.keys():
            self.data[k] = dict()
            self.data[k]['update'] = update
        else:
            if self.data[k]['update'] != update:
                raise ValueError

        if update:
            use_step = v['use_step'] if 'use_step' in v.keys() else True

            if 'y' not in self.data[k].keys():
                self.data[k]['x'] = [total_steps if use_step else epoch]
                self.data[k]['y'] = [v['y']]
            else:
                self.data[k]['x'].append(total_steps if use_step else epoch)
                self.data[k]['y'].append(v['y'])
        else:
            self.data[k]['x'] = v['x']
            self.data[k]['y'] = v['y']

    def visdom_visualize(self, k, vis):
        if self.data[k]['update']:
            y = np.array([self.data[k]['y'][-1]])
            x = np.array([self.data[k]['x'][-1]])
            u = None if len(self.data[k]['x']) == 1 else 'append'
        else:
            y = self.data[k]['y']
            x = self.data[k]['x']
            u = None
        vis.line(Y=y, X=x, win=k, opts=self.data[k]['opts'], update=u)

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                vis.line(Y=v['y'], X=v['x'], win=k, opts=v['opts'], update=None)

    def matplotlib_visualize(self, k, epoch, total_steps):
        plt.plot(self.data[k]['x'], self.data[k]['y'])
        if self.data[k]['update']:
            plt.savefig(os.path.join(self.save_path, '{}.png'.format(k)))
        else:
            plt.savefig(os.path.join(self.save_path, '{}_{:0>6d}_{:0>9d}.png'.format(k, epoch, total_steps)))
        plt.close()


class VISImage(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISImage, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        self.data.clear()

        self.data[k] = dict()

        if isinstance(v['images'], np.ndarray):
            img = v['images']
        else:
            tmp = v['images']
            tmp = [np.array(t).astype(np.float)[np.newaxis, ...] for t in tmp]
            tmp = np.concatenate(tmp, axis=0)
            img = tmp.transpose((0, 3, 1, 2)) / 255

        self.data[k]['images'] = img
        self.data[k]['nrow'] = v['nrow'] if 'nrow' in v.keys() else 1

    def visdom_visualize(self, k, vis):
        vis.images(self.data[k]['images'], win=k, opts=self.data[k]['opts'], nrow=self.data[k]['nrow'])

    def restore(self, vis):
        pass

    def matplotlib_visualize(self, k, epoch, total_steps):
        nrow = self.data[k]['nrow']
        if nrow > self.data[k]['images'].shape[0]:
            nrow = self.data[k]['images'].shape[0]

        save_img = []
        rows = math.ceil(self.data[k]['images'].shape[0] / nrow)
        for r in range(rows):
            row_img = []
            for j in range(nrow):
                i = r * nrow + j
                if i < self.data[k]['images'].shape[0]:
                    if self.data[k]['images'].max() <= 1:
                        mu = 255
                    else:
                        mu = 1
                    row_img.append((self.data[k]['images'][i].transpose((1, 2, 0)) * mu))
                else:
                    row_img.append(np.zeros(self.data[k]['images'][0].transpose((1, 2, 0)).shape))
                row_img.append(np.zeros((self.data[k]['images'].shape[2], 5, 3)))
            row_img = np.hstack(row_img[:-1])
            save_img.append(row_img)
            save_img.append(np.zeros((5, row_img.shape[1], 3)))

        save_img = np.vstack(save_img[:-1]).astype(np.uint8)

        image_pil = Image.fromarray(save_img)
        image_pil.save(os.path.join(self.save_path, '{}_{:0>6d}_{:0>9d}.jpg'.format(k, epoch, total_steps)))


class VISText(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISText, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        if k not in self.data:
            self.data[k] = dict()
        self.data[k]['text'] = v['text']

    def visdom_visualize(self, k, vis):
        text = self.data[k]['text'].replace('\n', '<br>')
        vis.text(text, win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                text = v['text'].replace('\n', '<br>')
                vis.text(text, win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        with open(os.path.join(self.save_path, '{}_{:0>6d}_{:0>9d}.txt'.format(k, epoch, total_steps)), 'w') as f:
            f.write(self.data[k]['text'])


class VISHeatmap(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISHeatmap, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        # assert v['x'].max() <= 1
        if k not in self.data:
            self.data[k] = dict()
        self.data[k]['x'] = v['x']

    def visdom_visualize(self, k, vis):
        vis.heatmap(X=self.data[k]['x'], win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                vis.heatmap(X=v['x'], win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        pass
        # fig, ax = plt.subplots(1, 1)
        # c = ax.pcolor(v['x'])
        # fig.colorbar(c, ax=ax)
        # plt.savefig(os.path.join(self.save_path, 'heatmap', '{}_{:0>6d}_{:0>9d}.png'.format(k, epoch, total_steps)))
        # plt.close()


class VISHistogram(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISHistogram, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        if k not in self.data:
            self.data[k] = dict()
        self.data[k]['x'] = v['x']

    def visdom_visualize(self, k, vis):
        vis.histogram(X=self.data[k]['x'], win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                vis.histogram(X=v['x'], win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        pass


class VISBar(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISBar, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        if k not in self.data:
            self.data[k] = dict()
        self.data[k]['x'] = v['x']

    def visdom_visualize(self, k, vis):
        vis.bar(X=self.data[k]['x'], win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                vis.bar(X=v['x'], win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        pass


class VISScatter(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISScatter, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        if k not in self.data:
            self.data[k] = dict()
        self.data[k]['x'] = v['x']
        self.data[k]['y'] = v['y']

    def visdom_visualize(self, k, vis):
        vis.scatter(Y=self.data[k]['y'], X=self.data[k]['x'], win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                vis.scatter(Y=v['y'], X=v['x'], win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        pass


class VISBoxplot(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISBoxplot, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        update = v['update'] if 'update' in v.keys() else False

        if k not in self.data.keys():
            self.data[k] = dict()
            self.data[k]['update'] = update
        else:
            if self.data[k]['update'] != update:
                raise ValueError

        if update:
            if 'x' not in self.data[k].keys():
                self.data[k]['x'] = np.array(v['x'])
                assert self.data[k]['x'].ndim == 0
            else:
                self.data[k]['x'] = np.vstack([self.data[k]['x'], v['x']])
        else:
            self.data[k]['x'] = v['x']

    def visdom_visualize(self, k, vis):
        if np.size(self.data[k]['x']) == 1:
            temp = np.vstack([self.data[k]['x'], self.data[k]['x']])
        else:
            temp = self.data[k]['x']
        vis.boxplot(X=temp, win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                if np.size(v['x']) == 1:
                    temp = np.vstack([v['x'], v['x']])
                else:
                    temp = v['x']
                vis.boxplot(X=temp, win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        pass
        # plt.plot(self.data[k]['x'], self.data[k]['y'])
        # if self.data[k]['update']:
        #     plt.savefig(os.path.join(self.save_path, '{}.png'.format(k)))
        # else:
        #     plt.savefig(os.path.join(self.save_path, '{}_{:0>6d}_{:0>9d}.png'.format(k, epoch, total_steps)))
        # plt.close()


class VISWafer(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISWafer, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        if k not in self.data:
            self.data[k] = dict()
        self.data[k]['x'] = v['x']

    def visdom_visualize(self, k, vis):
        vis.pie(X=self.data[k]['x'], win=k, opts=self.data[k]['opts'])

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                vis.pie(X=v['x'], win=k, opts=v['opts'])

    def matplotlib_visualize(self, k, epoch, total_steps):
        pass


class VISStackedLines(VIS):
    def __init__(self, save_path, use_visdom):
        super(VISStackedLines, self).__init__(save_path, use_visdom)

    def parse_data(self, k, v, epoch, total_steps):
        update = not 'x' in v.keys()

        if k not in self.data.keys():
            self.data[k] = dict()
            self.data[k]['update'] = update
        else:
            if self.data[k]['update'] != update:
                raise ValueError

        if update:
            use_step = v['use_step'] if 'use_step' in v.keys() else True

            if v['y'].ndim != 1:
                raise ValueError
            if v['y'].shape[0] < 2:
                raise ValueError

            if 'y' not in self.data[k].keys():
                self.data[k]['x'] = np.array([total_steps if use_step else epoch])
                self.data[k]['y'] = v['y'][np.newaxis, ...]
            else:
                self.data[k]['x'] = np.concatenate([self.data[k]['x'], np.array([total_steps if use_step else epoch])], axis=0)
                self.data[k]['y'] = np.concatenate([self.data[k]['y'], v['y'][np.newaxis, ...]], axis=0)
        else:
            self.data[k]['x'] = v['x']
            self.data[k]['y'] = v['y']

    def visdom_visualize(self, k, vis):
        if self.data[k]['update']:
            y = self.data[k]['y'][-1][np.newaxis, ...]
            x = np.repeat(self.data[k]['x'][-1], y.shape[1])[np.newaxis, ...]
            u = None if len(self.data[k]['x']) == 1 else 'append'
        else:
            y = self.data[k]['y']
            x = self.data[k]['x']
            u = None

        y = np.cumsum(y, axis=1)
        self.data[k]['opts']['fillarea'] = True
        vis.line(Y=y, X=x, win=k, opts=self.data[k]['opts'], update=u)

    def restore(self, vis):
        if self.use_visdom:
            for k, v in self.data.items():
                y = self.data[k]['y']
                
                if v['update']:
                    x = np.repeat(self.data[k]['x'][..., np.newaxis], y.shape[1], axis=1)
                else:
                    x = self.data[k]['x']

                y = np.cumsum(y, axis=1)
                self.data[k]['opts']['fillarea'] = True
                vis.line(Y=y, X=x, win=k, opts=self.data[k]['opts'], update=None)

    def matplotlib_visualize(self, k, epoch, total_steps):
        if self.data[k]['update']:
            x = np.repeat(self.data[k]['x'][..., np.newaxis], self.data[k]['y'].shape[1], axis=1)
        else:
            x = self.data[k]['x']

        y = np.cumsum(self.data[k]['y'], axis=1)
        y0 = np.zeros((y.shape[0], 1))
        y = np.concatenate([y0, y], axis=1)

        for i in range(x.shape[1]):
            plt.plot(x[..., i], y[..., i + 1])
            plt.fill_between(x[..., i], y[..., i + 1], y[..., i])

        plt.savefig(os.path.join(self.save_path, '{}.png'.format(k)))
        plt.close()


class Visualizer(object):
    def __init__(self, cfg):
        assert not cfg.test

        save_path = os.path.join(cfg.general.checkpoints_dir, cfg.general.experiment_name)
        self.use_visdom = True if cfg.visualizer.visdom else False

        self.vtgs = dict(
            line=VISLine(save_path, self.use_visdom),
            image=VISImage(save_path, self.use_visdom),
            text=VISText(save_path, self.use_visdom),
            heatmap=VISHeatmap(save_path, self.use_visdom),
            histogram=VISHistogram(save_path, self.use_visdom),
            bar=VISBar(save_path, self.use_visdom),
            scatter=VISScatter(save_path, self.use_visdom),
            boxplot=VISBoxplot(save_path, self.use_visdom),
            wafer=VISWafer(save_path, self.use_visdom),
            stackedlines=VISStackedLines(save_path, self.use_visdom),
        )

        if self.use_visdom:
            import visdom
            if '_' in cfg.general.experiment_name:
                print('实验名\"{}\"中有下划线，导致visdom中的本实验的enviornment被分为两级'.format(cfg.general.experiment_name))
            username = cfg.visualizer.visdom.username if cfg.visualizer.visdom.username else None
            password = cfg.visualizer.visdom.password if cfg.visualizer.visdom.password else None
            self.vis = visdom.Visdom(server=cfg.visualizer.visdom.server,
                                     port=cfg.visualizer.visdom.port,
                                     env=cfg.general.experiment_name,
                                     username=username,
                                     password=password)
            if not self.vis.check_connection():
                raise NotImplementedError('无法连接visdom，请检查网络设置以及visdom是否已启动')
            self.env_name = cfg.general.experiment_name

    def visualize(self, vis_data, epoch, total_steps):
        for k, v in vis_data.items():
            if k == 'progress':
                self.visualize_progress(v, epoch, total_steps)
                continue

            self.vtgs[k].visualize(v, epoch, total_steps, self.vis)

        if self.use_visdom:
            self.vis.save([self.env_name])

    def visualize_progress(self, od, epoch, total_steps):
        if not self.use_visdom:
            return

        epoch_fm, epoch_prefix = od['epoch']
        if len(epoch_prefix) > 0:
            prefix = epoch_prefix + '<br>'
        else:
            prefix = ''

        if 'step' in od.keys():
            step_fm, step_prefix = od['step']
            res_str = prefix + decode_format_dict(epoch_fm) + '<br>' + step_prefix + '<br>' + decode_format_dict(step_fm) + '<br>current_steps: {}'.format(total_steps)
        else:
            res_str = prefix + decode_format_dict(epoch_fm) + '<br>current_steps: {}'.format(total_steps)
            
        self.vis.text(res_str, win='progress', opts=dict(title='progress'))

    def get_vis_state_dict(self):
        fields = dict()
        for k, v in self.vtgs.items():
            fields[k] = v.get_data()
        return fields

    def set_vis_state_dict(self, vis_data):
        for k, v in vis_data.items():
            self.vtgs[k].set_data(v)
            self.vtgs[k].restore(self.vis)

        if self.use_visdom:
            self.vis.save([self.env_name])
