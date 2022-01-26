#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for Akasha
"""

import os


system = os.uname().sysname
open_cmd = 'open' if system == 'Darwin' else 'xdg-open'
