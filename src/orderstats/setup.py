from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cythonized",            # Change this to your module name
        sources=["cythonized.pyx"],  # Change this to your Cython file
        include_dirs=[np.get_include()],
        language="c"
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)
