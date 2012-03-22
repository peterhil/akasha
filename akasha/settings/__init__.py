#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import locale
import numpy as np
import sys


def setup():
    np.set_printoptions(precision=16, threshold=1000, edgeitems=10, linewidth=78, suppress=True)

    # Set the user's default locale, see http:// docs.python.org/library/locale.html
    # Also be sure to have LC_ALL='fi_FI.UTF-8' and CHARSET='UTF-8' set in the environment
    # to have sys.stdin.encoding = UTF-8
    locale.setlocale(locale.LC_ALL, 'fi_FI.UTF-8')
    assert sys.stdin.encoding == 'UTF-8', \
    	"Unicode not enabled! Current input encoding is: %s" % sys.stdin.encoding

setup()