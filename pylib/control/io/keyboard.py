#!/usr/bin/env python
# encoding: utf-8
"""
keyboard.py

Created by Peter on 2011-12-06.
Copyright (c) 2011 Loihde. All rights reserved.
"""

from __future__ import absolute_import

import json
import numpy as np
import os

from ... import settings


#  E1234567890123 456 789E
#  §1234567890+´B FHA N=/*
#  Tqwertyuiopå¨R DEV 789-
#  Lasdfghjklöä'R     456+
# S<zxcvbnm,.-SSS  ^  123E
# COM ______MMOCC <v> 00,E

#  E1234567890123 456 789E
#   §1234567890+´B FHA N=/*
#    Tqwertyuiopå¨R DEV 789-
#     Lasdfghjklöä'R     456+
#     S<zxcvbnm,.-SSS  ^  123E
#     COMM______MMOCC <v> 00,E


# Defined by:
# - Mapping from key to position
# - Mapping from position to frequency (by index)

def get_layout(path='settings/keymaps/fi.json', mapdir=settings.basedir):
    mappath = os.path.abspath('/'.join([mapdir, path]))
    fp = None
    try:
        fp = open(mappath)
        data = json.load(fp, encoding='utf-8')
        return data
    except Exception, e:
        raise e
    finally:
        if hasattr(fp, 'close'):
            fp.close()

def get_mapping(layout, section='main', mapping=np.empty([6,25], dtype=object)):
    if section == 'main':
        basecol = 0
    elif section == 'arrows':
        basecol = 14
    elif section == 'keypad':
        basecol = 17
    else:
        basecol = 0
    kbsect = layout[section]
    for row in xrange(len(kbsect)):
        for col in xrange(len(kbsect[row])):
            key = kbsect[row][col]
            mapping[row, col+basecol] = key
            #print "({0:d}, {1:d}) = {2!s}".format(row, col+basecol, key)
    return mapping

def get_keyboard(layout=get_layout()):
    kb = np.empty([6,25], dtype=object)
    kb.fill({})
    kb = get_mapping(layout, 'main', kb)
    kb = get_mapping(layout, 'arrows', kb)
    kb = get_mapping(layout, 'keypad', kb)
    return kb

def get_map(kb, key='key'):
    mp = {}
    cols = kb.shape[1]
    for i in xrange(kb.size):
        code = kb.item(i).get(key)
        if code:
            mp[code] = divmod(i, cols)
        else:
            mp[code] = kb.shape # Key disabled
    return mp

kb = get_keyboard()
pos = get_map(kb, 'key')
scan = get_map(kb, 'scancode')
