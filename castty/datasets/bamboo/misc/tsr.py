import os
import math
import torch
import numpy as np
from ..builder import INTERNODE
from PIL import Image, ImageDraw
from ..base_internode import BaseInternode
from ...utils.common import get_image_size


__all__ = ['CalcTSRGT']


@INTERNODE.register_module()
class CalcTSRGT(BaseInternode):
    def __init__(self, split=16, ratio=0.2, tau=0.25, **kwargs):
        assert split > 1
        assert 0 < ratio < 0.5
        assert 0 < tau < 1

        self.split = split
        self.ratio = ratio
        self.tau = tau

    def calc_points_on_x(self, input_points, w):
        points = []
        ind = []

        sid = np.argsort(input_points[:, 0])

        for p in input_points[sid]:
            if len(points) == 0:
                points.append(p)
                ind.append(True)
            else:
                if (p == points[-1]).all():
                    continue
                else:
                    points.append(p)
                    ind.append(True)

        if points[0][0] > 0:
            points = [np.array([0, points[0][1]]).astype(np.float32)] + points
            ind = [False] + ind
        if points[-1][0] < w - 1:
            points.append(np.array([w + 1, points[-1][1]]).astype(np.float32))
            ind[-1] = False
            ind.append(False)
        return np.array(points).astype(np.float32), np.array(ind).astype(np.bool_)

    def calc_points_on_y(self, input_points, h):
        points = []
        ind = []

        sid = np.argsort(input_points[:, 1])

        for p in input_points[sid]:
            if len(points) == 0:
                points.append(p)
                ind.append(True)
            else:
                if (p == points[-1]).all():
                    continue
                else:
                    points.append(p)
                    ind.append(True)

        if points[0][1] > 0:
            points = [np.array([points[0][0], 0]).astype(np.float32)] + points
            ind = [False] + ind
        if points[-1][1] < h - 1:
            points.append(np.array([points[-1][0], h + 1]).astype(np.float32))
            ind.append(False)
        return np.array(points).astype(np.float32), np.array(ind).astype(np.bool_)

    def calc_x(self, col, y, ind=None):
        i = np.nonzero(col[:, 1] <= y)[0][-1]

        if i == len(col) - 1:
            if ind is None:
                return col[i, 0]
            else:
                return col[i, 0], ind[i]

        x1, y1 = col[i]
        x2, y2 = col[i + 1]
        if y2 - y1 == 0:
            x = x2
        else:
            x = ((x2 - x1) / (y2 - y1)) * (y - y1) + x1

        if ind is None:
            return x
        else:
            return x, ind[i]

    def calc_y(self, row, x, ind=None):
        i = np.nonzero(row[:, 0] <= x)[0][-1]

        if i == len(row) - 1:
            if ind is None:
                return row[i, 1]
            else:
                return row[i, 1], ind[i]

        x1, y1 = row[i]
        x2, y2 = row[i + 1]
        if x2 - x1 == 0:
            y = y2
        else:
            y = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1

        if ind is None:
            return y
        else:
            return y, ind[i]

    def forward(self, data_dict, **kwargs):
        w, h = get_image_size(data_dict['image'])

        # ------------生成行GT部分------------

        rows = []
        rows_ind = []

        for table_id in np.unique(data_dict['bbox_meta']['table_id']):
            startrows = data_dict['bbox_meta']['startrow'][data_dict['bbox_meta']['table_id'] == table_id]

            ind = (data_dict['bbox_meta']['startrow'] == 0) & (data_dict['bbox_meta']['table_id'] == table_id)
            points = data_dict['point'][ind][:, [0, 1], :].reshape(-1, 2)
            points, ind = self.calc_points_on_x(points, w)
            rows.append(points)
            rows_ind.append(ind)

            # for i in range(len(np.unique(data_dict['bbox_meta']['startrow']))):
            for i in range(len(np.unique(startrows))):
                ind = (data_dict['bbox_meta']['endrow'] == i) & (data_dict['bbox_meta']['table_id'] == table_id)
                points = data_dict['point'][ind][:, [2, 3], :].reshape(-1, 2)
                points, ind = self.calc_points_on_x(points, w)
                rows.append(points)
                rows_ind.append(ind)
            # print(table_id, np.unique(startrows))
        # exit()

        row_separators = [[], rows, []]

        for i in range(len(rows)):
            ut = []
            bt = []
            for j in range(len(rows[i])):
                if i == 0:
                    d1 = abs(self.calc_y(rows[i + 1], rows[i][j, 0]) - rows[i][j, 1])
                    d2 = abs(0 - rows[i][j, 1])
                elif i == len(rows) - 1:
                    d1 = abs(h - rows[i][j, 1])
                    d2 = abs(self.calc_y(rows[i - 1], rows[i][j, 0]) - rows[i][j, 1])
                else:
                    d1 = abs(self.calc_y(rows[i + 1], rows[i][j, 0]) - rows[i][j, 1])
                    d2 = abs(self.calc_y(rows[i - 1], rows[i][j, 0]) - rows[i][j, 1])
                distance = min(d1, d2) * self.ratio

                ut.append([rows[i][j, 0], rows[i][j, 1] - distance])
                bt.append([rows[i][j, 0], rows[i][j, 1] + distance])

            ut = np.array(ut).astype(np.float32)
            bt = np.array(bt).astype(np.float32)

            row_separators[0].append(ut)
            row_separators[2].append(bt)

        # from castty.utils.point_tools import draw_point, draw_point_without_label
        # from PIL import ImageFont
        # img = data_dict['image'].copy()

        # w, h = img.size
        # l = math.sqrt(h * h + w * w)
        # draw = ImageDraw.Draw(img)
        # r = max(2, int(l / 200))
        # font = ImageFont.truetype('/Users/liaya/Documents/part-of-hitogata/castty/fonts/arial.ttf', 20)

        # # for row_separator in row_separators:
        # for i in range(len(rows_ind)):
        #     # i = 0
        #     row, row_ind = row_separators[1][i], rows_ind[i]
        #     for j, point in enumerate(row):
        #         x, y = np.around(point).astype(np.int32)
        #         if row_ind[j]:
        #             draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
        #         else:
        #             draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 255))
        #         draw.text((x, y), str(j), fill=(100, 255, 100), font=font)

        #     for j in range(len(row) - 1):
        #         x1, y1 = np.around(row[j]).astype(np.int32)
        #         x2, y2 = np.around(row[j + 1]).astype(np.int32)
        #         if row_ind[j] and row_ind[j + 1]:
        #             draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 0, 0))
        #         else:
        #             draw.line((x1, y1, x2, y2), width=r // 2, fill=(0, 0, 255))
        #     # break
        # img.show()
        # exit()

        x_tau = self.tau * w
        row_refpt = torch.zeros(h).type(torch.float32)

        for i in range(len(row_separators[0])):
            y1_tau = self.calc_y(row_separators[0][i], x_tau)
            y2_tau = self.calc_y(row_separators[1][i], x_tau)
            y3_tau = self.calc_y(row_separators[2][i], x_tau)

            radius = (y3_tau - y1_tau) / 2
            y2_tau = int(y2_tau)
            radius = int(radius) if int(radius) > 0 else 1

            ys = torch.arange(h).type(torch.float32)
            sigma = math.sqrt(0.5 * radius ** 2 / math.log(10))
            r = -(ys - y2_tau) ** 2 / (2 * sigma ** 2)
            heat = r.exp()
            heat[heat < 0.1] = 0

            row_refpt += heat

        xs = np.linspace(0, w, num=self.split + 1)[1:-1]
        gt_l_row = []
        gt_c_row = []

        for i in range(len(row_separators[0])):
            t = []
            c = []
            for x in xs:
                upper = self.calc_y(row_separators[0][i], x)
                middle, pind = self.calc_y(row_separators[1][i], x, rows_ind[i])
                lower = self.calc_y(row_separators[2][i], x)

                t.append([upper, middle, lower])
                c.append(pind)
            gt_l_row.append(t)
            gt_c_row.append(c)

        gt_l_row = np.array(gt_l_row).astype(np.float32)
        gt_c_row = np.array(gt_c_row).astype(np.bool_)
        # print(gt_l_row, gt_l_row.shape)
        # print(gt_c_row, gt_c_row.shape)
        # exit()

        row_mask = Image.new('P', (w, h), 0)
        draw = ImageDraw.Draw(row_mask)

        for i in range(len(row_separators[0]) - 1):
            polygon = np.concatenate([row_separators[2][i], row_separators[0][i + 1][::-1]])
            draw.polygon(polygon.astype(np.int32).flatten().tolist(), fill=1)

        gt_row_mask = np.array(row_mask)

        # ------------生成列GT部分------------

        cols = []
        cols_ind = []

        for table_id in np.unique(data_dict['bbox_meta']['table_id']):
            startcols = data_dict['bbox_meta']['startcol'][data_dict['bbox_meta']['table_id'] == table_id]

            ind = (data_dict['bbox_meta']['startcol'] == 0) & (data_dict['bbox_meta']['table_id'] == table_id)
            points = data_dict['point'][ind][:, [0, 3], :].reshape(-1, 2)
            points, ind = self.calc_points_on_y(points, h)
            cols.append(points)
            cols_ind.append(ind)

            # for i in range(len(np.unique(data_dict['bbox_meta']['startcol']))):
            for i in range(len(np.unique(startcols))):
                ind = (data_dict['bbox_meta']['endcol'] == i) & (data_dict['bbox_meta']['table_id'] == table_id)
                points = data_dict['point'][ind][:, [1, 2], :].reshape(-1, 2)
                points, ind = self.calc_points_on_y(points, h)
                cols.append(points)
                cols_ind.append(ind)

        col_separators = [[], cols, []]

        for i in range(len(cols)):
            lt = []
            rt = []
            for j in range(len(cols[i])):
                if i == 0:
                    d1 = abs(self.calc_x(cols[i + 1], cols[i][j, 1]) - cols[i][j, 0])
                    d2 = abs(0 - cols[i][j, 0])
                elif i == len(cols) - 1:
                    d1 = abs(w - cols[i][j, 0])
                    d2 = abs(self.calc_x(cols[i - 1], cols[i][j, 1]) - cols[i][j, 0])
                else:
                    d1 = abs(self.calc_x(cols[i + 1], cols[i][j, 1]) - cols[i][j, 0])
                    d2 = abs(self.calc_x(cols[i - 1], cols[i][j, 1]) - cols[i][j, 0])
                distance = min(d1, d2) * self.ratio

                lt.append([cols[i][j, 0] - distance, cols[i][j, 1]])
                rt.append([cols[i][j, 0] + distance, cols[i][j, 1]])

            lt = np.array(lt).astype(np.float32)
            rt = np.array(rt).astype(np.float32)

            col_separators[0].append(lt)
            col_separators[2].append(rt)

        # from castty.utils.point_tools import draw_point, draw_point_without_label
        # from PIL import ImageFont
        # img = data_dict['image'].copy()

        # w, h = img.size
        # l = math.sqrt(h * h + w * w)
        # draw = ImageDraw.Draw(img)
        # r = max(2, int(l / 200))
        # font = ImageFont.truetype('/Users/liaya/Documents/part-of-hitogata/castty/fonts/arial.ttf', 20)

        # # for row_separator in row_separators:
        # for i in range(len(cols_ind)):
        #     # i = 0
        #     col, col_ind = col_separators[1][i], cols_ind[i]
        #     for j, point in enumerate(col):
        #         x, y = np.around(point).astype(np.int32)
        #         if col_ind[j]:
        #             draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
        #         else:
        #             draw.ellipse((x - r, y - r, x + r, y + r), fill=(0, 0, 255))
        #         draw.text((x, y), str(j), fill=(100, 255, 100), font=font)

        #     for j in range(len(col) - 1):
        #         x1, y1 = np.around(col[j]).astype(np.int32)
        #         x2, y2 = np.around(col[j + 1]).astype(np.int32)
        #         if col_ind[j] and col_ind[j + 1]:
        #             draw.line((x1, y1, x2, y2), width=r // 2, fill=(255, 0, 0))
        #         else:
        #             draw.line((x1, y1, x2, y2), width=r // 2, fill=(0, 0, 255))
        #     # break
        # img.show()
        # exit()

        y_tau = self.tau * h
        col_refpt = torch.zeros(w).type(torch.float32)

        for i in range(len(col_separators[0])):
            x1_tau = self.calc_x(col_separators[0][i], y_tau)
            x2_tau = self.calc_x(col_separators[1][i], y_tau)
            x3_tau = self.calc_x(col_separators[2][i], y_tau)

            radius = (x3_tau - x1_tau) / 2
            x2_tau = int(x2_tau)
            radius = int(radius) if int(radius) > 0 else 1
            # print(x2_tau, radius)

            xs = torch.arange(w).type(torch.float32)
            sigma = math.sqrt(0.5 * radius ** 2 / math.log(10))
            r = -(xs - x2_tau) ** 2 / (2 * sigma ** 2)
            heat = r.exp()
            heat[heat < 0.1] = 0

            col_refpt += heat

        ys = np.linspace(0, h, num=self.split + 1)[1:-1]
        gt_l_col = []
        gt_c_col = []

        for i in range(len(col_separators[0])):
            t = []
            c = []
            for y in ys:
                left = self.calc_x(col_separators[0][i], y)
                middle, pind = self.calc_x(col_separators[1][i], y, cols_ind[i])
                right = self.calc_x(col_separators[2][i], y)

                t.append([left, middle, right])
                c.append(pind)
            gt_l_col.append(t)
            gt_c_col.append(c)

        gt_l_col = np.array(gt_l_col).astype(np.float32)
        gt_c_col = np.array(gt_c_col).astype(np.bool_)

        # print(gt_l_col, gt_l_col.shape)
        # print(gt_c_col, gt_c_col.shape)

        col_mask = Image.new('P', (w, h), 0)
        draw = ImageDraw.Draw(col_mask)

        for i in range(len(col_separators[0]) - 1):
            polygon = np.concatenate([col_separators[2][i], col_separators[0][i + 1][::-1]])
            draw.polygon(polygon.astype(np.int32).flatten().tolist(), fill=1)
        # img = Image.blend(img, col_mask, 0.5)
        # img.show()

        gt_col_mask = np.array(col_mask)

        data_dict['tsr_l_row'] = torch.from_numpy(gt_l_row) / h
        data_dict['tsr_c_row'] = torch.from_numpy(gt_c_row)
        data_dict['tsr_row_mask'] = torch.from_numpy(gt_row_mask)
        data_dict['tsr_row_refpt'] = row_refpt

        data_dict['tsr_l_col'] = torch.from_numpy(gt_l_col) / w
        data_dict['tsr_c_col'] = torch.from_numpy(gt_c_col)
        data_dict['tsr_col_mask'] = torch.from_numpy(gt_col_mask)
        data_dict['tsr_col_refpt'] = col_refpt

        # exit()
        return data_dict

    def backward(self, data_dict):
        return data_dict

    def __repr__(self):
        return 'CalcTSRGT(split={}, ratio={})'.format(self.split, self.ratio)

