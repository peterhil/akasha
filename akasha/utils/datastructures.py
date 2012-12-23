#!/usr/bin/env python
# -*- coding: utf-8 -*-


def chaff(dic, pred=lambda x: x < 5):
    """
    Weed out (pun intended) some items in a dictionary, returning those items
    and modifying the dictionary.

    >>> foo = {'foo': 4, 'bar': 6, 'quuz': 7, 'boo': 2}
    >>> res = chaff(foo)
    >>> print("Keep: {0}\nDiscard: {1}".format(foo, res))
    Keep: {'quuz': 7, 'bar': 6}
    Discard: {'foo': 4, 'boo': 2}
    """
    wheat = {}
    keys_to_remove = [k for (k, v) in dic.iteritems() if pred(v)]
    for k in keys_to_remove:
        wheat[k] = dic.pop(k)
    return wheat
