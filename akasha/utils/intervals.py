#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interval tree module.
"""

from __future__ import absolute_import

from builtins import range

from akasha.utils.python import class_name


class Interval:
    """
    Interval.
    """

    def __init__(self, inf, sup):
        self._inf = inf
        self._sup = sup

    @property
    def inf(self):
        """
        Infimum of the interval.
        """
        return self._inf

    @property
    def sup(self):
        """
        Supremum of the interval.
        """
        return self._sup

    def __repr__(self):
        return f'{class_name(self)}({self.inf}, {self.sup})'


class IntervalTree:
    """
    Interval tree.

    This is a modified port of this BSD Licensed Ruby implementation of
    augmented interval tree:
    https://github.com/misshie/interval-tree/blob/master/lib/interval_tree.rb

    The code this is modified from a Python port by Tyler Kahn:
    http://forrst.com/posts/Interval_Tree_implementation_in_python-e0K
    """

    def __init__(self, intervals):
        self.top_node = self.divide_intervals(intervals)

    def divide_intervals(self, intervals):
        """
        Divide intervals into subtrees.
        """
        if not intervals:
            return None

        x_center = self.center(intervals)

        s_center = []
        s_left = []
        s_right = []

        for k in intervals:
            if k.sup < x_center:
                s_left.append(k)
            elif k.inf > x_center:
                s_right.append(k)
            else:
                s_center.append(k)

        return Node(
            x_center,
            s_center,
            self.divide_intervals(s_left),
            self.divide_intervals(s_right),
        )

    def center(self, intervals):
        """ """
        fs = self.sort_by_inf(intervals)

        return fs[int(len(fs) / 2)].inf

    def search(self, begin, end=None):
        """ """
        if end:
            result = []

            for j in range(begin, end + 1):
                for k in self.search(j):
                    result.append(k)
                result = list(set(result))
            return self.sort_by_inf(result)
        else:
            return self._search(self.top_node, begin, [])

    def _search(self, node, point, result):
        """ """
        for k in node.s_center:
            if k.inf <= point <= k.sup:
                result.append(k)
        if point < node.x_center and node.left_node:
            for k in self._search(node.left_node, point, []):
                result.append(k)
        if point > node.x_center and node.right_node:
            for k in self._search(node.right_node, point, []):
                result.append(k)

        return list(set(result))

    @staticmethod
    def sort_by_inf(intervals):
        """
        Sort intervals by their infimum (lower limits).
        """
        return sorted(intervals, key=lambda x: x.inf)


class Node:
    """
    Interval tree node.
    """

    def __init__(self, x_center, s_center, left_node, right_node):
        self.x_center = x_center
        self.s_center = IntervalTree.sort_by_inf(s_center)
        self.left_node = left_node
        self.right_node = right_node
