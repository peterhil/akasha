"""
Logging for Akasha.

Copyright (c) 2011 Peter. All rights reserved.

Author: Peter Hillerström
Date: 2011-12-06
"""

import sys
import logging
from akasha.settings import config


class ansi:
    """
    ANSI colour codes
    """

    # pylint: disable=W1401

    # TODO refactor using code() like in Prism...
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'

    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'

    DIM_BLACK = '\033[30m'
    DIM_RED = '\033[31m'
    DIM_GREEN = '\033[32m'
    DIM_YELLOW = '\033[33m'

    DIM_BLUE = '\033[34m'
    DIM_MAGENTA = '\033[35m'
    DIM_CYAN = '\033[36m'
    DIM_WHITE = '\033[37m'

    BRIGHT = '\033[1m'
    RESET = '\033[0m'
    END = NORMAL = RESET

    # Logging

    NOTSET = RESET
    ANIMA = DIM_WHITE
    BORING = BLACK
    DEBUG = CYAN
    INFO = BRIGHT + GREEN
    WARNING = WARN = YELLOW
    ERROR = BRIGHT + RED
    CRITICAL = FATAL = DIM_RED

    @classmethod
    def color(cls, name):
        """
        Usage: ansi.color('red')
        """
        # TODO: API like ansi.color('DEBUG', ansi.BRIGHT)
        return cls.__dict__[name.upper()]

    def disable(self):
        """
        Disable all colors
        """
        self.BLACK = ''
        self.RED = ''
        self.GREEN = ''
        self.YELLOW = ''
        self.BLUE = ''
        self.MAGENTA = ''
        self.CYAN = ''
        self.WHITE = ''
        self.RESET = ''


class ColorFormatter(logging.Formatter):
    """
    Color log formatter, that maps log levels to colors.
    """

    def format(self, record):
        record.color = ansi.color(record.__dict__['levelname'])
        return super().format(record)


logging.ANIMA = 4
logging.addLevelName(logging.ANIMA, 'ANIMA')

logging.BORING = 5
logging.addLevelName(logging.BORING, 'BORING')

logger = logging.getLogger('Akasha')

absformatter = ColorFormatter(
    "%(color)s%(asctime)s %(name)s: [%(levelname)s]\t%(message)s" + ansi.END
)

relformatter = ColorFormatter(
    "%(color)s%(relativeCreated)12.4f %(name)s: "
    "[%(levelname)s]\t%(message)s" + ansi.END
)

handler = logging.StreamHandler(sys.stderr)
# handler = logging.FileHandler('/var/log/akasha.log')

handler.setFormatter(relformatter)
logger.addHandler(handler)
logger.setLevel(config.logging_limits.LOGLEVEL)
