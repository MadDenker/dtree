#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from distutils.core import setup, Command # pylint: disable=no-name-in-module

import dtree

class TestCommand(Command):
    description = "Runs unittests."
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system('python dtree_pd.py')

setup(
    name='dtree_pd',
    version=dtree_pd.__version__,
    description='A Python decision tree using Pandas DataFrames.',
    author='Lucas Finco',
    author_email='lmfinco@uwalumni.com',
    url='https://github.com/MadDenker/dtree_pd',
    license='LGPL',
    py_modules=['dtree_pd'],
    ###https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Data Scientists",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    platforms=['OS Independent'],
#    test_suite='dtree_pd',
    cmdclass={
        'test': TestCommand,
    },
)
