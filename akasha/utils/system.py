#!/usr/bin/env python

import os


system = os.uname().sysname
open_cmd = 'open' if system == 'Darwin' else 'xdg-open'
