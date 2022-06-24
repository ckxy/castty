import cv2
import math
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

matplotlib.use('Agg')


class Gauge(object):
    def __init__(self, data_mode, num_range, ang_range, colors='jet_r'):
        self.data_mode = data_mode
        if self.data_mode == 'me_6s1p':
            self.n = 6
        else:
            raise ValueError

        if isinstance(colors, str):
            cmap = cm.get_cmap(colors, self.n - 1)
            cmap = cmap(np.arange(self.n - 1))
            self.colors = cmap[::-1,:].tolist()
        if isinstance(self.colors, list): 
            if len(self.colors) == self.n - 1:
                self.colors = self.colors[::-1]
            else: 
                raise Exception("\n\nnumber of colors {} not equal \
                to number of categories{}\n".format(len(self.colors), self.n - 1))

        assert num_range[1] >= num_range[0]
        assert ang_range[1] >= ang_range[0]

        start = np.linspace(ang_range[0], ang_range[1], self.n, endpoint=True)[0:-1]
        end = np.linspace(ang_range[0], ang_range[1], self.n, endpoint=True)[1::]
        self.ang_range = np.c_[start, end]
        self.ticks = [self.ang_range[0][0]] + self.ang_range[..., 1].tolist()

        coors = []
        for a in self.ticks[::-1]:
            x1 = math.cos(a * math.pi / 180)
            y1 = -math.sin(a * math.pi / 180)
            coors.append((x1, y1))
        self.coors = np.array(coors)

        self.labels = np.linspace(num_range[0], num_range[1], self.n).tolist()[::-1]

    def calc_value(self, in_p, warped_center=False):
        center = None

        if self.data_mode == 'me_6s1p':
            points, tip = in_p[:6], in_p[6]
            if warped_center:
                center = in_p[7]

        M, _ = cv2.findHomography(points, self.coors)

        p = tip.reshape(1, 1, 2)
        p = cv2.perspectiveTransform(p, M).squeeze()

        if center is not None:
            c = tip.reshape(1, 1, 2)
            c = cv2.perspectiveTransform(c, M).squeeze()
        else:
            c = np.zeros(2)

        raduis = np.linalg.norm(c - p)
        angle = math.acos((p[0] - c[0]) / raduis) * 180 / math.pi

        s = self.ticks[-1] - self.ticks[0]
        r = (angle - self.ticks[0]) / s
        v = (1 - r) * self.labels[0]

        return v, self.draw_gauge(angle, v)

    def rot_text(self, ang): 
        return np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))

    def draw_gauge(self, angle, value): 
        """
        begins the plotting
        """
        
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        
        """
        plots the sectors and the arcs
        """
        patches = []
        for ang, c in zip(self.ang_range, self.colors): 
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))
        
        [ax.add_patch(p) for p in patches]

        
        """
        set the labels (e.g. 'LOW','MEDIUM',...)
        """


        for t, lab in zip(self.ticks, self.labels): 
            ax.text(0.25 * np.cos(np.radians(t)), 0.25 * np.sin(np.radians(t)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=10, \
                fontweight='bold', rotation=self.rot_text(t))

        # """
        # set the bottom banner and the title
        # """
        
        ax.text(0, -0.05, '{:.5f}'.format(value), horizontalalignment='center', \
             verticalalignment='center', fontsize=22, fontweight='bold')

        # """
        # plots the arrow now
        # """
        
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(angle)), 0.225 * np.sin(np.radians(angle)), \
                     width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
        
        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        """
        removes frame and ticks, and makes axis equal and tight
        """
        
        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas.draw()       # draw the canvas, cache the renderer
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close()

        return image


# def agauge(points, num_range, ang_range, angle, colors='jet_r'): 
#     pass


if __name__ == '__main__':
    # gauge(points=6, num_range=(0, 2.5), ang_range=(-45, 225), angle=151, colors='YlOrRd_r') 
    g = Gauge(n=6, num_range=(0, 2.5), ang_range=(-45, 225))
    # g.gauge(angle=151, colors='YlOrRd_r')
    plt.show()