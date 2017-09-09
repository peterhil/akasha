#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Graphics output module
"""

import os
import pylab as lab
import tempfile

from PIL import Image

from akasha.graphic.drawing import draw
from akasha.utils.log import logger


__all__ = ['graph', 'imsave', 'show']


def show(img, plot=False, osx_open=False):
    """
    Show an image from a Numpy array.
    """
    img = img.transpose((1, 0, 2))
    if plot:
        lab.interactive(True)
        imgplot = lab.imshow(img)
        imgplot.set_cmap('hot')
        lab.show(False)
    elif osx_open:
        try:
            tmp = tempfile.NamedTemporaryFile(dir='/var/tmp', prefix='akasha_', suffix='.png', delete=False)
            logger.debug("Tempfile: %s" % tmp.name)
            image = Image.fromarray(img)
            image.save(tmp.name, 'png')
            os.system("open " + tmp.name)
        except IOError, err:
            logger.error("Failed to open a temporary file and save the image: %s" % err)
        except OSError, err:
            logger.error("Failed to open the image with a default app: %s" % err)
        finally:
            tmp.close()
    else:
        image = Image.fromarray(img)
        image.show()


def imsave(img, filename):
    try:
        img = img.transpose((1, 0, 2))
        image = Image.fromarray(img)
        image.save(filename, 'png')
    except IOError, err:
        logger.error("Failed to save image into {}".format(filename))
    except Exception, err:
        logger.error("Error when saving image: {}".format(err))


def graph(signal, size=1000, plot=False, axis=True,
          antialias=True, lines=False, colours=True, img=None, osx_open=False):
    """
    Make an image from the sound signal and show it.
    """
    img = draw(
        signal,
        size=size,
        antialias=antialias, lines=lines, colours=colours,
        axis=axis,
        img=img
    )
    show(img, plot, osx_open)
