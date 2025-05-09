"""
Graphics drawing module.
"""

import pygame
import numpy as np

from PIL import Image
# from builtins import range
from skimage import draw as skdraw

from akasha.funct.itertools import pairwise
from akasha.graphic.colour import colorize, white
from akasha.utils.array import is_silence
from akasha.utils.log import logger
from akasha.math import (
    as_pixels,
    clip,
    complex_as_reals,
    flip_vertical,
    get_pixels,
    get_points,
    inside,
    normalize,
    pad,
    pcm,
    roundcast,
)


def draw_axis(img, colour=None):
    """Draw axis on the image with the colour."""
    if colour is None:
        colour = [42, 42, 42, 127]
    height, width, channels = img.shape
    img[height // 2, :] = img[:, width // 2] = colour[:channels]

    return img


def get_canvas(width=1000, height=None, channels=4):
    """Get a Numpy array suitable for use as a drawing canvas."""
    if height is None:
        height = width

    img = np.zeros((height, width, channels), np.uint8)  # Note: y, x

    return img


def draw_blank(img):
    """Draw blank image."""
    black = [0, 0, 0, 255]
    height, width, channels = img.shape
    img[::] = black[:channels]

    return img


def blit(screen, img):
    """Blit the screen."""
    if screen and img is not None:
        pygame.surfarray.blit_array(screen, img[..., :3])  # Drop alpha


def draw(
    signal,
    size=1000,
    antialias=False,
    lines=False,
    colours=True,
    axis=True,
    img=None,
    screen=None,
):
    """Draw the complex sound signal into specified size image."""
    # See: http://jehiah.cz/archive/creating-images-with-numpy

    # TODO: Buffering with frame rate for animations or
    # realtime signal view. For example: 44100 Hz / 30 Hz = 1470 samples
    # buffersize = int(round(float(sampler.rate) / framerate))
    # sl = slice(item.start, item.stop, buffersize).indices(item.stop)
    # indices = np.arange(*sl)
    # for start in indices:
    #     stop = start + buffersize - 1
    #     signal = self[start : stop : buffersize]

    # TODO: Enable using non-square size.
    if img is not None:  # Draw into existing img?
        size = img.shape[0]
    else:
        img = get_canvas(size)

    if axis:
        img = draw_axis(img)

    if is_silence(signal):
        logger.warning('Drawing empty signal!')
        return img

    signal = clip_samples(signal)

    if lines:
        if screen is None:
            logger.warning(
                "Drawing lines with Numpy is way too slow for now!"
            )
            img = draw_lines(signal, img, size, colours, antialias)
        else:
            img = draw_lines_pg(signal, screen, img, size, colours, antialias)
    else:
        if antialias:
            img = draw_points_aa(signal, img, size, colours)
        else:
            img = draw_points(signal, img, size, colours)

    return img


def clip_samples(signal):
    """Clip a signal into unit rectangle area."""
    clip_max = np.max(np.fmax(np.abs(signal.real), np.abs(signal.imag)))
    if clip_max > 1.0:
        logger.warning(
            "Clipping signal -- maximum magnitude was: %0.6f", clip_max
        )
        return clip(signal)
    return signal


def add_alpha(rgb, opacity=255):
    """Add alpha channel with specified opacity to the rgb signal."""
    alpha = np.array([opacity] * len(rgb), dtype=np.uint8).reshape(
        len(rgb), 1
    )

    return np.append(rgb, alpha, 1)


def draw_lines_pg(
    signal, screen, img, size=1000, colours=True, antialias=False
):
    """Draw (antialiased) lines with Pygame."""
    draw_blank(img)
    img = draw_axis(img)
    blit(screen, img)

    method = 'aaline' if antialias else 'line'
    pts = get_points(flip_vertical(signal), size).T
    if colours:
        # FIXME: aaline draws wrong colours on high frequencies!
        colors = add_alpha(colorize(signal))
        for (i, ends) in enumerate(pairwise(pts)):
            getattr(pygame.draw, method)(screen, colors[i], *ends)
    else:
        getattr(pygame.draw, method + 's')(
            screen, pygame.Color('orange'), False, pts, 1
        )
    return None


def draw_lines(signal, img, size=1000, colours=True, antialias=False):
    """Draw antialiased lines with Numpy."""
    if len(signal) < 2:
        signal = pad(signal, -1)
    if colours:
        colors = add_alpha(colorize(signal))

    points = (
        np.rint(get_points(flip_vertical(signal), size) - 0.5)
        .astype(np.uint32)
        .T
    )
    segments = np.hstack((points[:-1], points[1:]))

    try:
        for i, coords in enumerate(segments):
            if antialias:
                [x, y, values] = skdraw.line_aa(*coords)
                color = colors[i] if colours else white

                # Use alpha values from lines_aa
                color_values = np.repeat(
                    color[:, np.newaxis], len(values), axis=1
                ).T
                color_values[:, 3] = (values * color_values[:, 3]).astype(
                    np.uint8
                )

                img[x, y] += color_values
            else:
                img[skdraw.line(*coords)] = colors[i] if colours else white
    except (IndexError, ValueError):
        import ipdb

        ipdb.set_trace()

    return img


def draw_points_np_aa(signal, img, size=1000, colours=True):
    """Draw colourized antialiased points from signal."""
    points = (clip(signal) + 1 + 1j) / 2.0 * (size - 1) + (0.5 + 0.5j)
    deltas = points - np.round(points)

    color = add_alpha(colorize(signal)) if colours else white

    # TODO: Should these offsets to deltas vary?
    offset = -0.5 - 0.5j
    ints = np.asarray(complex_as_reals(deltas + offset), dtype=np.int32) * color

    img[:-1, :-1, :] = ints
    img[:-1, :-1, :] = ints
    img[:-1, :-1, :] = ints
    img[:-1, :-1, :] = ints

    return img


def draw_points_aa(signal, img, size=1000, colours=True):
    """Draw colourized antialiased points from signal."""
    # Fixme: Ignores size argument as it is now
    width, height, _ = img.shape

    def iw(a):
        return inside(a, 0, width)

    def ih(a):
        return inside(a, 0, height)

    def pixels(values):
        return roundcast(
            colors * as_pixels(values), dtype=np.uint8
        )

    px, value = get_pixels(signal, width - 1)
    colors = add_alpha(colorize(signal)) if colours else white

    img[px[0], px[1], :] += pixels(value[0] * value[1])
    img[px[0], iw(px[1] + 1), :] += pixels(value[0] * (1 - value[1]))
    img[ih(px[0] + 1), px[1], :] += pixels((1 - value[0]) * value[1])
    img[ih(px[0] + 1), iw(px[1] + 1), :] += pixels(
        (1 - value[0]) * (1 - value[1])
    )

    return img


def draw_points_aa_old(signal, img, size=1000, colours=True):
    """Draw colourized antialiased points from signal."""
    size -= 1
    bases, deltas = get_pixels(signal, size)

    values_00 = deltas[1] * deltas[0]
    values_01 = deltas[1] * (1.0 - deltas[0])
    values_10 = (1.0 - deltas[1]) * deltas[0]
    values_11 = (1.0 - deltas[1]) * (1.0 - deltas[0])

    pos = [
        ((size - 1) - bases[1], bases[0]),
        ((size - 1) - bases[1], bases[0] + 1),
        ((size - 1) - (bases[1] + 1), bases[0]),
        ((size - 1) - (bases[1] + 1), bases[0] + 1),
    ]

    color = add_alpha(colorize(signal)) if colours else white

    def pixels(values):
        return roundcast(color * as_pixels(values), dtype=np.uint8)

    img[pos[0][1], pos[0][0], :] += pixels(values_11)
    img[pos[1][1], pos[1][0], :] += pixels(values_10)
    img[pos[2][1], pos[2][0], :] += pixels(values_01)
    img[pos[3][1], pos[3][0], :] += pixels(values_00)

    return img


def draw_points(signal, img, size=1000, colours=True):
    """Draw colourized points from signal"""
    points = get_points(flip_vertical(signal), size) - 0.5
    points = np.rint(points).astype(np.uint32)

    color = add_alpha(colorize(signal)) if colours else white

    img[points[0], points[1]] = color[..., : img.shape[2]]
    return img


def hist_graph(signal, size=1000):
    """Use Numpy histogram2d to make an image from
    the complex signal.
    """
    # See Numpy docs on numpy.histogram2d:
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram2d.html
    #
    # TODO: see np.bincount and Numpy histogram with 3x3d arrays as indices
    # https://stackoverflow.com/questions/7422713/numpy-histogram-with-3x3d-arrays-as-indices
    x, y = get_points(flip_vertical(signal))
    histogram, _, _ = np.histogram2d(
        y, x, bins=size, range=[[-1.0, 1.0], [-1.0, 1.0]], normed=True
    )
    histogram = normalize(histogram)
    image = Image.fromarray(np.array(histogram * 255, dtype=np.uint8), 'L')
    image.show()
    return histogram


def video_transfer(signal, standard='PAL', axis='real', horiz=720):
    """Draw a sound signal using the old video tape audio recording
    technique. See:
    http://en.wikipedia.org/wiki/44100_Hz#Recording_on_video_equipment
    """
    # PAL:
    # 294 × 50 × 3 = 44,100
    # 294 active lines/field × 50 fields/second × 3 samples/line =
    #     44,100 samples/second
    # (588 active lines per frame, out of 625 lines total)

    # NTSC:
    # 245 × 60 × 3 = 44,100
    # 245 active lines/field × 60 fields/second × 3 samples/line =
    #     44,100 samples/second
    # (490 active lines per frame, out of 525 lines total)

    # See 576i and 576p: horiz. 720 or 704, vert. 576 out of 625 lines
    formats = {
        'PAL': 588,
        'NTSC': 490,
    }
    vert = formats[standard]

    linewidth = 3  # samples per line
    framesize = vert * linewidth  # 1764 for PAL, 1470 for NTSC

    # Make complex signal real
    if isinstance(signal[0], np.complex):
        signal = getattr(signal, axis)

    # for block in range(0, len(signal), framesize):
    #    pass # draw frame

    img = get_canvas(3, vert)  # Stretch to horiz. width later!
    fv = img.flat

    s = pcm(signal[:framesize] * 256, bits=8, axis='real').astype(np.uint8)
    # fv[::] = np.repeat(signal[:framesize].T, 4, axis=0).T  # y, x
    fv[::] = np.repeat(s, 4, axis=0)

    # logger.debug(
    #     "Image:\n%s,\nFlat view:\n%s",
    #     img[:framesize], fv[:framesize]
    # )

    return np.repeat(img[:framesize], horiz / linewidth + 2, axis=1)
