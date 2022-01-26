#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Numpy setup
"""

import locale
import numpy as np
import six
import sys


def np_setup():
    """
    Setup numpy and locale settings.
    """
    np.set_printoptions(precision=16, threshold=1000, edgeitems=40, linewidth=78, suppress=True)

    # Set the user's default locale, see http:// docs.python.org/library/locale.html
    # Also be sure to have LC_ALL='fi_FI.UTF-8' and CHARSET='UTF-8' set in the environment
    # to have sys.stdin.encoding = UTF-8
    locale.setlocale(locale.LC_ALL, 'fi_FI.UTF-8')

    assert locale.getlocale()[1] in ('UTF8', 'UTF-8'), \
        "Unicode not enabled! Current locale is: %s.%s" % locale.getlocale()

    if six.PY2 and isinstance(sys.stdin, file):
        assert sys.stdin.encoding == 'UTF-8', \
            "Unicode input not enabled! Current input encoding is: %s" % sys.stdin.encoding
