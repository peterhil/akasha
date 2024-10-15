#!/usr/bin/env python

"""
Akasha audio program suite
"""


import os

from akasha.settings import config, np_setup
from akasha.utils.log import logger


np_setup()


if not hasattr(config, 'basedir'):
    config.basedir = os.path.abspath(os.path.dirname(__file__))
    logger.info('Started from: %s', config.basedir)
