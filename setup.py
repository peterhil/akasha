#!/usr/bin/env python
# encoding: utf-8
#
# Copyright (c) 2009-2017, Peter Hillerström <peter.hillerstrom@gmail.com>
# All rights reserved.

from __future__ import with_statement

import sys
from setuptools import setup, Command

PACKAGE_NAME = 'akasha'
PACKAGE_VERSION = '0.1.0'

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
        errno = subprocess.call([sys.executable, '-m', 'pytest', 'akasha/test'])
        raise SystemExit(errno)

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    packages=[
        'akasha',
        'akasha.audio',
        'akasha.audio.envelope',
        'akasha.audio.mixins',
        'akasha.control',
        'akasha.control.io',
        'akasha.curves',
        'akasha.dsp',
        'akasha.effects',
        'akasha.funct',
        'akasha.graphic',
        'akasha.graphic.primitive',
        'akasha.math',
        'akasha.math.geometry',
        'akasha.net',
        'akasha.settings',
        'akasha.types',
        'akasha.utils',
    ],
    package_data={
        'akasha.settings': ['keymaps/*.json'],
    },
    install_requires = [
        'Pillow (>=4.2.1)',
        'm3-cdecimal (>=2.3)',
        'funckit (==0.8.0)',
        'ipython (>=5.4.1)',
        'matplotlib (>=1.4.2)',
        'numpy (>=1.11.3)',
        'ordereddict (>=1.1)' if sys.version_info < (2, 7) else '',
        'pandas (>=0.20.1)',
        'pygame (>=1.9.2)',
        'scikit-image (>=0.11.2)',
        'scikits.audiolab (>=0.11.0)',
        'scikits.samplerate (>=0.3.3)',
        'scipy (>=0.19.0)',
        'wikitools (==1.1.1)',
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
