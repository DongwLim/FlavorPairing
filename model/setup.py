from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="Cython_dataset", 
    ext_modules=cythonize("cython_dataset.pyx"), 
    include_dirs=[numpy.get_include()]
)