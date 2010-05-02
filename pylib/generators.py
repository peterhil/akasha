#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from timing import Sampler
from PIL import Image


class Generator:

    def __getitem__(self, item):
        """Slicing support."""
        if isinstance(item, slice):
            # Construct an array of indices.
            item = np.arange(*(item.indices(item.stop)))
        return self.sample(item)
    
    def graph(self, item, size=600, framerate=30):
        """Graph of the complex sound signal."""
        # See http://jehiah.cz/archive/creating-images-with-numpy
        buffersize = int(round(float(Sampler.rate) / framerate))    # 44100.0/30 = 1470
        indices = np.arange(*(slice(item.start, item.stop, buffersize).indices(item.stop)))
        img = np.zeros((size,size,4), np.uint8) # Note: y, x
        # for start in indices:
        # samples = self[start:start+buffersize-1:buffersize] # TODO: Make this cleaner
        
        samples = self[item]
        # Scale to size
        samples = (samples + 1+1j) / 2.0 * (size - 1)
        # Convert complex samples to pairs of reals
        coords = samples.view(np.float64).reshape(len(samples), 2).transpose()
        # Cast to integers
        coords = np.cast['uint64'](coords)
        img[size/2.0,:] = img[:,size/2.0] = [51,204,204,255]    # Draw axis
        img[size - coords[1], coords[0]] = [255,255,255,255]
        image = Image.fromarray(img, 'RGBA')
        image.show()


class PeriodicGenerator(Generator):

    def __getitem__(self, item):
        """Slicing support. If given a slice the behaviour will be:

        # Step defaults to 1, is wrapped modulo period, and can't be zero!
        # Start defaults to 0, is wrapped modulo period
        # Number of elements returned is the absolute differerence of 
        # stop - start (or period and 0 if either value is missing)
        # Element count is multiplied with step to produce the same 
        # number of elements for different step values.
        """
        if isinstance(item, slice):
            step = ((item.step or 1) % self.period or 1)
            start = ((item.start or 0) % self.period)
            element_count = abs((item.stop or self.period) - (item.start or 0))
            stop = start + (element_count * step)
            # Construct an array of indices.
            item = np.arange(*(slice(start, stop, step).indices(stop)))
            # print item[-1] % self.period # Could be used as cursor
        return self.samples[np.array(item) % self.period]

    def __len__(self):
        return self.period