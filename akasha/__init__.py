#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Akasha audio program suite
"""

from __future__ import absolute_import

import os

from akasha.settings import config, np_setup
from akasha.utils.log import logger


np_setup()


if not hasattr(config, 'basedir'):
    config.basedir = os.path.abspath(os.path.dirname(__file__))
    logger.info('Started from: %s', config.basedir)
