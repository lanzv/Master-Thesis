from distutils.core import setup
from Cython.Build import cythonize
from numpy.distutils.core import setup
import numpy


setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm/random_utils.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm/chant.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm/npylm.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm/hyperparameters.pyx"),
    include_dirs=[numpy.get_include()],
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm/blocked_gibbs_sampler.pyx"),
    include_dirs=[numpy.get_include()],
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm/viterbi_algorithm.pyx"),
    packages=['nhpylm']
)
setup(
    name='nhpylm model',
    ext_modules=cythonize("src/models/nhpylm_model.pyx"),
    include_dirs=["src/models/nhpylm/", numpy.get_include()]
)