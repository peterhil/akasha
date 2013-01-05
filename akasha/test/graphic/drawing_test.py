#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C0111: Missing docstring
# R0201: Method could be a function
# E1101: Module 'x' has no 'y' member
#
# pylint: disable=C0111,R0201,E1101
"""
Unit tests for drawing functions
"""

import numpy as np
import pytest

# from akasha.graphic.drawing import graph


class TestGraph(object):
    """Tests for graph()"""

    @pytest.skip("Incomplete")
    def test_graph_coord_bounds(self):
        """Test coordinate boundaries"""

        ### Make complex signal containing grid points

        x_axis = y_axis = np.linspace(-1, 1, 40)

        # In [213]: x_axis
        # Out[213]: array([-1.    , -0.3333,  0.3333,  1.    ])
        #
        # In [214]: y_axis
        # Out[214]: array([-1.    , -0.3333,  0.3333,  1.    ])
        #

        grid = np.array(np.meshgrid(x_axis, y_axis))

        # In [281]: grid
        # Out[281]:
        # array([[[-1.    , -0.3333,  0.3333,  1.    ],
        #         [-1.    , -0.3333,  0.3333,  1.    ],
        #         [-1.    , -0.3333,  0.3333,  1.    ],
        #         [-1.    , -0.3333,  0.3333,  1.    ]],
        #
        #        [[-1.    , -1.    , -1.    , -1.    ],
        #         [-0.3333, -0.3333, -0.3333, -0.3333],
        #         [ 0.3333,  0.3333,  0.3333,  0.3333],
        #         [ 1.    ,  1.    ,  1.    ,  1.    ]]])

        samples = grid.copy().transpose().view(np.complex).flatten()

        # In [286]: samples
        # Out[286]:
        # array([-1.-1.j    , -1.-0.3333j, -1.+0.3333j, ...,  1.-0.3333j,
        #         1.+0.3333j,  1.+1.j    ])

        # graph(samples)
