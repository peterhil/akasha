#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
import PIL.Image


# Code from: http://mentat.za.net/numpy/intro/intro.html#the-misunderstood-for-loop


ITERATIONS = 100
DENSITY = 1000 # warning: execution speed decreases with square of DENSITY

x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

x, y = np.meshgrid(np.linspace(x_min, x_max, DENSITY),
                   np.linspace(y_min, y_max, DENSITY))

c = x + 1j*y # complex grid
z = c.copy()
fractal = np.zeros(z.shape, dtype=np.uint8) + 255

for n in range(ITERATIONS):
    print "Iteration %d" % n

    # --- Uncomment to see different sets ---

    # Tricorn
    # z = z.conj()

    # Burning ship
    # z = abs(z.real) + 1j*abs(z.imag)

    # ---

    # Leave the lines below in place
    z *= z
    z += c

    mask = (fractal == 255) & (abs(z) > 10)
    fractal[mask] = 254 * n / float(ITERATIONS)

# plt.imshow(np.log(fractal), cmap=plt.cm.hot,
#            extent=(x_min, x_max, y_min, y_max))
# plt.title('Mandelbrot Set')
# plt.xlabel('Re(z)')
# plt.ylabel('Im(z)')
# plt.show()

f = PIL.Image.fromarray(fractal)
f.save('images/fractal.png')

