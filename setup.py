import os
import logging
from Cython.Build import cythonize
import numpy
from setuptools import setup, find_packages
from distutils.core import Extension

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')


def get_requirements():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, "requirements.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


PACKAGE_NAME = 'part-of-hitogata'
VERSION = '0.0.1'
AUTHOR = 'ckxy'

if __name__ == '__main__':
    logger.info(f"Installing {PACKAGE_NAME} (v: {VERSION}) ...")
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        licence_file='LICENCE',
        packages=find_packages(),
        install_requires=get_requirements(),
        include_package_data=True,
        # python_requires='~=3.7',
        ext_modules=cythonize(Extension(
            'part_of_hitogata.datasets.bamboo.misc.pse.pse',
            sources=['part_of_hitogata/datasets/bamboo/misc/pse/pse.pyx'],
            language='c++',
            include_dirs=[numpy.get_include()],
            library_dirs=[],
            libraries=[],
            extra_compile_args=['-O3'],
            extra_link_args=[]
            )
        )
    )