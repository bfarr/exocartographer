#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import re

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

version_re = re.compile("__version__ = \"(.*?)\"")
with open(path.join(path.dirname(path.abspath(__file__)), "exocartographer", "__init__.py")) as inp:
    r = inp.read()
version = version_re.findall(r)[0]

setup(
    name='exocartographer',
    version=version,
    description='Map some exoplanets',
    author='Ben Farr, Will Farr, Nick Cowan',
    author_email='bfarr@uoregon.edu, w.farr@bham.ac.uk, nicolas.cowan@mcgill.ca',
    url='https://github.com/bfarr/exocartographer',
    include_package_data=True,
    packages=['exocartographer'],
    install_requires=['numpy', 'healpy', 'scipy'],
)
