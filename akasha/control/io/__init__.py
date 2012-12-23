#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from akasha import settings


def relative_path(path, base=settings.basedir):
    return os.path.abspath('/'.join([base, path]))
