#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Graphics output module
"""

import os
import tempfile
import pylab as lab

from PIL import Image

from akasha.graphic.drawing import draw
from akasha.utils import open_cmd
from akasha.utils.log import logger


__all__ = ['graph', 'imsave', 'show']


def show(img, plot=False, use_open=False):
    """
    Show an image from a Numpy array.
    """
    img = img.transpose((1, 0, 2))
    if plot:
        lab.interactive(True)
        imgplot = lab.imshow(img)
        imgplot.set_cmap('hot')
        lab.show(False)
    elif use_open:
        try:
            tmp = tempfile.NamedTemporaryFile(
                dir='/var/tmp',
                prefix='akasha_',
                suffix='.png',
                delete=False
            )
            logger.debug("Tempfile: %s", tmp.name)
            image = Image.fromarray(img)
            image.save(tmp.name, 'png')
            os.system(' '.join([open_cmd, tmp.name]))
        except IOError as err:
            logger.error("Failed to open a temporary file and save the image: %s", err)
        except OSError as err:
            logger.error("Failed to open the image with a default app: %s", err)
        finally:
            tmp.close()
    else:
        image = Image.fromarray(img)
        image.show()


def imsave(img, filename):
    """
    Save the image into file.
    """
    try:
        img = img.transpose((1, 0, 2))
        image = Image.fromarray(img)
        image.save(filename, 'png')
    except IOError as err:
        logger.error("Failed to save image into file: '%s'\n\nError was: %s", filename, err)
    except OSError as err:
        logger.error("OS error when saving image: %s", err)
    finally:
        image.close()


def graph(signal, plot=False, use_open=True, **kwargs):
    """
    Make an image from the sound signal and show it.
    Accepts the same keyword arguments as draw().
    """
    img = draw(signal, **kwargs)
    show(img, plot, use_open)
