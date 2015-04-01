#!/usr/bin/env python
# encoding: utf-8
#
# Copyright (c) 2009-2012, Peter Hillerström <peter.hillerstrom@gmail.com>
# All rights reserved.

from __future__ import with_statement

import sys
# from distutils.core import setup, Command
from setuptools import setup, Command

PACKAGE_NAME = 'akasha'
PACKAGE_VERSION = '0.0.1-dev'

# with open('README.rst', 'r') as readme:
#     README_TEXT = readme.read()

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py', 'akasha/test'])
        raise SystemExit(errno)

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    packages=[
        'akasha',
        'akasha.analysis',
        'akasha.audio',
        'akasha.control',
        'akasha.control.io',
        'akasha.effects',
        'akasha.funct',
        'akasha.funct.xoltar',
        'akasha.graphic',
        'akasha.graphic.primitive',
        'akasha.net',
        'akasha.settings',
        'akasha.types',
        'akasha.utils',
    ],
    package_data={
        'akasha.settings': ['keymaps/*.json'],
    },
    install_requires=['distribute'],
    requires = [
        'PIL (==1.1.7)',
        'Twisted (==12.2.0)',
        'cdecimal (>=2.3)',
        'distribute (>=0.6.32)',
        'funckit (==0.8.0)',
        'ipython (>=0.13)',
        'matplotlib (>=1.1.1)',
        'numpy (>=1.6.2)',
        'ordereddict (>=1.1)' if sys.version_info < (2, 7) else 'collections',
        'pandas (>=0.16.0)',
        'py (>=1.4.11)',
        'pygame (>=1.9.2)',
        'scikits.audiolab (>=0.11.0)',
        'scikits-image (>=0.7.1)',
        'scikits.samplerate (>=0.3.3)',
        'scipy (>=0.10.0)',
        'txosc (>=0.2.0)',
        'wikitools (>=1.1.1)',
    ],
    # scripts=['bin/akasha'],

    description="Akasha Resonance",
    # long_description=README_TEXT,
    author='Peter Hillerström',
    author_email='peter.hillerstrom@gmail.com',
    license='Proprietary',
    url='http://composed.nu/peterhil/aanet/',

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Environment :: MacOS X :: Cocoa',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications'
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Other Audience',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Other OS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Artistic Software',
        'Topic :: Multimedia :: Graphics :: Presentation',
        'Topic :: Multimedia :: Graphics :: Viewers'
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Multimedia :: Sound/Audio :: Capture/Recording',
        'Topic :: Multimedia :: Sound/Audio :: Conversion',
        'Topic :: Multimedia :: Sound/Audio :: Editors',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'Topic :: Multimedia :: Sound/Audio :: Mixers',
        'Topic :: Multimedia :: Sound/Audio :: Players',
        'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis',
        'Topic :: Utilities',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: pygame',
        'Topic :: System :: Archiving :: Compression',
    ],
    cmdclass = {
        'test': PyTest
    },
)
