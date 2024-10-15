#!/usr/bin/env python
#
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=E1101

"""
Copyright (c) 2011 Peter Hillerström. All rights reserved.

Author: Peter Hillerström
Date: 2011-12-06
"""


import json
import numpy as np
import sys


from akasha.io.path import relative_path


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

def get_layout(path='settings/keymaps/fi.json'):
    with open(relative_path(path)) as keymap:
        if sys.version_info >= (3, 9, 0):
            return json.load(keymap)
        else:
            return json.load(keymap, encoding='utf-8')

def get_mapping(
    layout, section='main', mapping=np.empty([6, 25], dtype=object)
):
    if section == 'main':
        basecol = 0
    elif section == 'arrows':
        basecol = 14
    elif section == 'keypad':
        basecol = 17
    else:
        basecol = 0
    kbsect = layout[section]
    for row in range(len(kbsect)):
        for col in range(len(kbsect[row])):
            key = kbsect[row][col]
            mapping[row, col + basecol] = key
            # print("({0:d}, {1:d}) = {2!s}".format(row, col+basecol, key))
    return mapping


def get_keyboard(layout=get_layout()):
    kb = np.empty([6, 25], dtype=object)
    kb.fill({})
    kb = get_mapping(layout, 'main', kb)
    kb = get_mapping(layout, 'arrows', kb)
    kb = get_mapping(layout, 'keypad', kb)
    return kb


def get_map(kb, key='key'):
    mp = {}
    cols = kb.shape[1]
    for i in range(kb.size):
        code = kb.item(i).get(key)
        if code:
            mp[code] = divmod(i, cols)
        else:
            mp[code] = kb.shape  # Key disabled
    return mp


kb = get_keyboard()
pos = get_map(kb, 'key')
scan = get_map(kb, 'scancode')
