#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _super(self):
    return super(self.__class__, self)


def issequence(arg):
    """
    Checks if arg is a sequence.

    For discussion, see:
    http://stackoverflow.com/questions/1835018/
    python-check-if-an-object-is-a-list-or-tuple-but-not-string/1835259#1835259
    """
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))
