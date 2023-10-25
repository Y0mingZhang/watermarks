from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("src/*.pyx", language_level=3),
    include_dirs=[np.get_include()],
    packages=find_packages(),
)
