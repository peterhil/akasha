"""
Numpy setup
"""

import locale
import numpy as np
import sys


def np_setup():
    """
    Setup numpy and locale settings.
    """
    np.set_printoptions(
        precision=16,
        threshold=1000,
        edgeitems=40,
        linewidth=78,
        suppress=True,
    )

    # Set the user's default locale, see:
    # http:// docs.python.org/library/locale.html
    #
    # Also be sure to have LC_ALL='fi_FI.UTF-8' and CHARSET='UTF-8' set in the
    # environment to have sys.stdin.encoding = UTF-8
    locale.setlocale(locale.LC_ALL, 'fi_FI.UTF-8')

    current_locale = '.'.join(locale.getlocale())
    assert locale.getlocale()[1] in ('UTF8', 'UTF-8'), (
        'Unicode not enabled! Current locale is: ' + current_locale
    )
