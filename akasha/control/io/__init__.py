#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os

from akasha import settings


def relative_path(path, base=settings.basedir):
	return os.path.abspath('/'.join([base, path]))

