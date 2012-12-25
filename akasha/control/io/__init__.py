#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from akasha.settings import config


def relative_path(path, base=config.basedir):
    return os.path.abspath('/'.join([base, path]))
