import os
import cv2
from PIL import Image, ExifTags

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)


def read_image_paths(root_dir):
    images = []
    assert os.path.isdir(root_dir), '{}是一个无效的目录'.format(root_dir)

    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return sorted(images)


def read_image_pil(path):
    img = Image.open(path)

    try:
        exif = dict(img._getexif().items())

        if exif[0x0112] == 3:
            img = img.rotate(180, expand=True)
        elif exif[0x0112] == 6: 
            img = img.rotate(270, expand=True)
        elif exif[0x0112] == 8:
            img = img.rotate(90, expand=True)
    except:
        pass

    return img.convert('RGB')


def read_image_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
