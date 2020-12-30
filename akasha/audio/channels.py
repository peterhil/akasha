#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Audio channels module
"""

import numpy as np


def num_channels(data):
    return 1 if np.ndim(data) <= 1 else data.shape[1]
