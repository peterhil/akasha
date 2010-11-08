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

def fast_graph(samples, size=1000, framerate=30):
    """Graph of the complex sound signal."""
    # See http://jehiah.cz/archive/creating-images-with-numpy
    
    img = np.zeros((size,size,4), np.uint8) # Note: y, x
    
    # Clip complex samples inside unit square and
    # view as real number coordinate pairs.
    clipped = clip(samples)
    coords = clipped.view(np.float).reshape(len(samples), 2).transpose()
    
    # Scale to size
    samples = (samples + 1+1j) / 2.0 * (size - 1)
    
    # Cast to integers
    coords = np.cast['uint64'](coords)
    
    # Draw image
    img[size/2.0,:] = img[:,size/2.0] = [51,204,204,255]    # Draw axis
    img[size - coords[1], coords[0]] = [255,255,255,255]
    image = Image.fromarray(img, 'RGBA')
    image.show()

def graph(samples, size=1000, framerate=30, plot=False):
    """Graph of the complex sound signal."""
    # See http://jehiah.cz/archive/creating-images-with-numpy
    
    # TODO: Buffering with frame rate for animations or realtime signal view.
    # buffersize = int(round(float(Sampler.rate) / framerate))    # 44100.0/30 = 1470
    # indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
    # for start in indices:
    # samples = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner
    
    img = np.zeros((size+1,size+1,4), np.uint8)             # Note: y, x    
    img[size/2.0,:] = img[:,size/2.0] = [51,204,204,127]    # Draw axis

    # Scale to size and interpret values as pixel centers
    samples = ((clip(samples) + 1+1j) / 2.0 * (size - 1) + (0.5+0.5j))    # 0.5 to 599.5
    
    # Convert complex samples to real number coordinate points
    points = samples.view(np.float).reshape(len(samples), 2).transpose()    # 0.5 to 599.5
    
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
        
    # Cast floats to integers
    img = np.cast['uint8'](img)
    
    if (plot and plt):
        imgplot = plt.imshow(img[:,:,0])
        imgplot.set_cmap('hot')
    else:
        image = Image.fromarray(img, 'RGBA')
        image.show()

def plot(samples):
    "Plot samples using matplotlib"
    imgplot = plt.imshow([samples.real, samples.imag])
    imgplot.set_cmap('hot')
