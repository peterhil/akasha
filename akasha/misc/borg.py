#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Borg:
    """Borg pattern, see http://code.activestate.com/recipes/66531/
    All objects share the same state - you will be assimilated!
    """
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
