import os
import dominate
from dominate.tags import *


class HTML(object):
    def __init__(self, web_dir, title, reflesh=0, dif=True):
        self.title = title
        self.web_dir = web_dir
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)

        self.dif = dif
        if dif:
            self.img_dir = os.path.join(self.web_dir, 'images')
            if not os.path.exists(self.img_dir):
                os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if reflesh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(reflesh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, width=400):
        self.add_table()
        with self.t:
            with tr():
                for im, txt in zip(ims, txts):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            if self.dif:
                                path = os.path.join('images', im)
                            else:
                                path = im
                            with a(href=path):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('./', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
    html.add_images(ims, txts)

    html.add_header('22222')

    ims = []
    txts = []
    for n in range(3):
        ims.append('image1_%d.png' % n)
        txts.append('text2_%d' % n)
    html.add_images(ims, txts)
    html.save()
