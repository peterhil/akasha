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

from akasha.audio.oscillator import Osc
from akasha.graphic.drawing import *
from akasha.utils.log import logger
from akasha.test import assert_equal_image

from mock import patch
from numpy.testing.utils import assert_array_equal


class TestDrawing(object):
    """
    Unit tests for drawing module.
    """

    def test_draw_axis(self):
        img = get_canvas(width=4, height=3, channels=1, axis=False)
        o = [0]
        x = [92]
        assert_array_equal(
            np.array([
                [o, o, x, o],
                [x, x, x, x],
                [o, o, x, o],
            ]),
            draw_axis(img, colour=[x])
        )

    def test_draw_axis_rgba(self):
        img = get_canvas(width=3, height=3, channels=4, axis=False)
        o = [0, 0, 0, 0]
        x = [33, 42, 51, 192]
        assert_array_equal(
            np.array([
                [o, x, o],
                [x, x, x],
                [o, x, o],
            ]),
            draw_axis(img, colour=[x])
        )

    @pytest.mark.parametrize(('width', 'height', 'channels'), [
        [5, None, 4],
        [3, 4, 2],
        [2, 1, 3],
        [1, 0, 1],
    ])
    def test_get_canvas(self, width, height, channels):
        assert_array_equal(
            np.zeros([height if height is not None else width, width, channels]),
            get_canvas(width, height, channels, axis=False)
        )

    @pytest.mark.parametrize(('func', 'args'), [
        ['draw_lines_pg', {
            'lines': True,
            'antialias': True,
            'screen': True,
        }],
        ['draw_lines', {
            'lines': True,
            'antialias': False,
            'screen': None,
        }],
        ['draw_points_aa', {
            'lines': False,
            'antialias': True,
            'screen': None,
        }],
        ['draw_points', {
            'lines': False,
            'antialias': False,
            'screen': None,
        }],
    ])
    def test_draw(self, func, args):
        draw_defaults = {
            'size': 7,
            'antialias': False,
            'lines': False,
            'colours': True,
            'axis': True,
            'img': None,
            'screen': None,
        }

        signal = Osc.from_ratio(1, 3).sample
        d = draw_defaults.copy()
        d.update(args)

        with patch('akasha.graphic.drawing.clip_samples') as clip_mock:
            with patch('akasha.graphic.drawing.' + func) as mock:
                draw(signal, **d)
                assert clip_mock.called_once_with(signal)
                assert mock.called_once_with(signal, d['screen'], d['size'], d['colours'])

    def test_clip_samples(self):
        with patch('akasha.utils.log.logger.warn') as log:
            assert -1+1j == clip_samples(-3+4j)
            assert log.called_once_with("Clipping signal -- maximum magnitude was: 5.000000")

    def test_clip_samples_noop(self):
        o = Osc.from_ratio(1, 16)
        assert_array_equal(o.sample, clip_samples(o.sample))

    @pytest.mark.parametrize('rgb', [
        [
            [41, 42, 43, 127],
            [51, 52, 53, 127],
        ],
        [
            [23, 127],
            [24, 127],
        ],
    ])
    def test_add_alpha(self, rgb):
        assert_array_equal(
            np.array(rgb),
            add_alpha(np.array(rgb)[..., :-1], 127)
        )

    def test_draw_coloured_lines_aa(self):
        pass

    def test_draw_coloured_lines(self):
        pass

    def test_draw_points_np_aa(self):
        pass

    ### Test for point drawing functions

    @pytest.mark.parametrize(('palette'), [
        [[255], [0], [42]],
        [[255, 255, 255, 255], [  0,   0,   0,   0], [ 40,  41,  42, 127]],
    ])
    @pytest.mark.parametrize(('draw_func'), [
        draw_points,
        draw_points_coo,
    ])
    def test_draw_points(self, palette, draw_func):
        """
        All point drawing functions should draw correctly.
        """
        osc = Osc.from_ratio(1, 8)
        size = 7
        x, _, a = palette

        assert_equal_image(
            np.array([
                [x, _, _, x, _, _, x],
                [_, _, _, a, _, _, _],
                [_, _, _, a, _, _, _],
                [x, a, a, a, a, a, x],
                [_, _, _, a, _, _, _],
                [_, _, _, a, _, _, _],
                [x, _, _, a, _, _, _],
            ], dtype=np.uint8).transpose(1, 0, 2),
            draw_func(
                osc[:6] * 2,
                img=draw_axis(
                    get_canvas(size, channels=len(palette[0]), axis=False),
                    colour=palette[2]
                ),
                size=size,
                colours=False
            )
        )

    @pytest.mark.parametrize(('draw_func'), [
        draw_points,
        draw_points_coo,
        # draw_points_aa,
        # draw_points_aa_old,
    ])
    def test_draw_points_coordinates_should_match(self, draw_func):
        """
        Test that coordinates and origin are headed in the right orientation.
        """
        size = 9
        channels = 4
        c = np.repeat(np.arange(256)[..., np.newaxis], channels, 1)
        x, _ = [c[255], c[0]]

        assert_equal_image(
            np.array([
                [ _, _, _, _, _, _, _, _, x,],
                [ _, _, _, _, _, _, _, x, _,],
                [ _, _, _, _, _, _, x, _, _,],
                [ _, _, _, _, _, x, _, _, _,],
                [ _, _, _, _, x, _, _, _, _,],
                [ _, _, _, _, _, _, _, _, _,],
                [ _, _, _, _, _, _, _, _, _,],
                [ _, _, _, _, _, _, _, _, _,],
                [ _, _, _, _, _, _, _, _, _,],
            ], dtype=np.uint8).transpose(1, 0, 2),
            draw_func(
                np.linspace(0, 1+1j, 5, endpoint=True),
                img=get_canvas(size, channels=channels, axis=False),
                size=size,
                colours=False
            )
        )

    @pytest.mark.parametrize(('palette'), [
        # [[255], [0], [42]],
        [[255, 255, 255, 255], [  0,   0,   0,   0], [ 127,  127,  127, 127]],
    ])
    @pytest.mark.parametrize(('draw_func'), [
        draw_points_aa,
        # draw_points_np_aa,
        # draw_points_aa_old,
    ])
    def test_draw_points_antialiased(self, palette, draw_func):
        """
        All point drawing functions should draw correctly.
        """
        size = 9
        x, _, a = palette
        l = np.repeat(np.arange(256)[..., np.newaxis], len(palette[2]), 1)

        assert_equal_image(
            np.transpose(np.array([
                [  l[23],  l[54],      _,      _,      a,      _,      _,      _,      _],
                [  l[54], l[125],      _,      _,      a,  l[48],  l[16],      _,      _],
                [      _,      _,      _,  l[25], l[152],  l[143], l[48],      _,      _],
                [      _,      _,      _, l[102], l[229],      _,      _, l[128],      _],
                [      a,      a,      a,      a,      a,      a,      a, l[255],      a],
                [      _,      _,      _,      _,      a,      _,      _,      _,      _],
                [      _, l[107],  l[71],      _,      a,      _,      _,      _,      _],
                [      _,  l[46],  l[31],      _,      a,      _,      _,      _,      _],
                [      _,      _,      _,      _,      a,      _,      _,      _,      _],
                ], dtype=np.uint8), (1, 0, 2)),
            draw_func(
                np.array([1, 0.5+0.5j, 0.2j, -0.8+0.8j, -0.6-0.8j]),
                draw_axis(
                    get_canvas(size, channels=len(palette[0]), axis=False),
                    colour=palette[2]
                ),
                size=size,
                colours=False
            )
        )

    def test_hist_graph(self):
        pass

    def test_video_transfer(self):
        pass
