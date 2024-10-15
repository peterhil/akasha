#!/usr/bin/env python

import os

from akasha.settings import config


def relative_path(path, base=config.basedir):
    return os.path.abspath('/'.join([base, path]))


def file_extension(filename):
    return os.path.splitext(filename)[1][1:]
