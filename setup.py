#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

setup(
    name='exocartographer',
    version='1.0',
    description='Map some exoplanets',
    author='Ben Farr, Will Farr, Nick Cowan',
    author_email='bfarr@uoregon.edu, w.farr@bham.ac.uk, nicolas.cowan@mcgill.ca',
    url='https://github.com/bfarr/exocartographer',
    include_package_data=True,
    packages=['exocartographer'],
    install_requires=['numpy', 'healpy', 'scipy'],
)
