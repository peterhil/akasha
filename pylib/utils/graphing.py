#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from timing import Sampler
from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except:
    pass

def normalize(signal):
    return signal / np.max(np.abs(signal))  # FIXME ZeroDivision if max=0!

def clip(signal, inplace=False):
    """Clips complex samples to unit area (-1-1j, +1+1j)."""
    if not inplace:
        signal = signal.copy()
    reals = signal.view(np.float)
    np.clip(reals, a_min=-1, a_max=1, out=reals)    # Uses out=reals to transform in-place!
    return signal

def hist_graph(samples, size=1000):
    """Uses numpy histogram2d to make an image from complex signal."""
    c = samples.view(np.float64).reshape(len(samples), 2).transpose()
    x, y = c[0], -c[1]
    hist, y_edges, x_edges = np.histogram2d(y, x, bins=size, range=[[-1.,1.],[-1.,1.]], normed=False)
    image = Image.fromarray(np.array(hist / hist.mean() * 255, dtype=np.uint8),'L')
    image.show()

def get_canvas(size=1000):
    return np.zeros((size+1,size+1,4), np.uint8)             # Note: y, x    

def get_points(samples, size=1000):
    # Scale to size and interpret values as pixel centers
    samples = ((clip(samples) + 1+1j) / 2.0 * (size - 1) + (0.5+0.5j))      # 0.5 to 599.5
    
    # Convert complex samples to real number coordinate points
    return samples.view(np.float).reshape(len(samples), 2).transpose()    # 0.5 to 599.5    

def draw(samples, size=1000, antialias=True):
    """Draw the complex sound signal into specified size image."""
    # See http://jehiah.cz/archive/creating-images-with-numpy
    
    # TODO: Buffering with frame rate for animations or realtime signal view.
    # buffersize = int(round(float(Sampler.rate) / framerate))    # 44100.0/30 = 1470
    # indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
    # for start in indices:
    # samples = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner
    
    img = get_canvas(size)
    points = get_points(samples, size)

    img[size/2.0,:] = img[:,size/2.0] = [51,204,204,127]    # Draw axis

    if antialias:
        # Draw with antialising
        centers = np.round(points)  # 1.0 to 600.0
        bases = np.cast['uint64'](centers) - 1   # 0 to 599
        deltas = points - bases - 0.5
    
        values_00 = deltas[1] * deltas[0]
        values_01 = deltas[1] * (1.0 - deltas[0])
        values_10 = (1.0 - deltas[1]) * deltas[0]
        values_11 = (1.0 - deltas[1]) * (1.0 - deltas[0])
    
        img[(size-1) - bases[1], bases[0], :] += np.repeat((values_11 * 255), 4).reshape(len(samples),4)
        img[(size-1) - bases[1], bases[0]+1, :] += np.repeat((values_10 * 255), 4).reshape(len(samples),4)
        img[(size-1) - (bases[1]+1), bases[0], :] += np.repeat((values_01 * 255), 4).reshape(len(samples),4)
        img[(size-1) - (bases[1]+1), bases[0]+1, :] += np.repeat((values_00 * 255), 4).reshape(len(samples),4)
    else:
        # Cast floats to integers
        points = np.cast['uint64'](points)  # 0 to 599
    
        # Draw image
        img[(size - 1) - points[1], points[0]] = [255,255,255,255]
        
    return img


# Showing images

def show(img, plot=False):
    if (plot and plt):
        imgplot = plt.imshow(img[:,:,0])
        imgplot.set_cmap('hot')
    else:
        image = Image.fromarray(img, 'RGBA')
        image.show()
    return False

def fast_graph(samples, size=1000, plot=False):
    return graph(samples, size, plot, antialias=False)

def graph(samples, size=1000, plot=False, antialias=True):
    img = draw(samples, size, antialias)
    show(img, plot)
    return False

def plot(samples):
    "Plot samples using matplotlib"
    imgplot = plt.imshow([samples.real, samples.imag])
    imgplot.set_cmap('hot')
    return False
