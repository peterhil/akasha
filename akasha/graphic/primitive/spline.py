#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clothoid splines.
http://www.dgp.toronto.edu/~karan/papers/sbim2008mccrae.pdf
"""

from __future__ import division

import numpy as np
import scipy as sc

from cmath import rect

from akasha.audio.curves import Ellipse
from akasha.audio.oscillator import Osc
from akasha.funct import consecutive
from akasha.graphic.drawing import plt
from akasha.graphic.geometry import circumcircle_radius, is_collinear
from akasha.utils.math import abspowersign, as_complex, lambertw


def clothoid_erf(t):
    """Euler spiral defined using the error function:
    http://en.wikipedia.org/wiki/Fresnel_integral#Properties"""
    return np.sqrt(np.pi / 2) * ((1 - 1j) / 2) * sc.special.erfi(t * (1 + 1j) / np.sqrt(2))


def clothoid_gray_hg(t, exponent=2, norm=1):
    """
    Generalization by Gray (1997): http://mathworld.wolfram.com/CornuSpiral.html
    Defined by generalized hypergeometric functions.

    Curvature is t**n.

    TODO: Something is not right here...
    """
    @np.vectorize
    def sf(t, n, a):
        coeff = (a * t ** (n + 2)) / ((n + 1) * (n + 2))
        n2p1 = (2 * (n + 1))
        a = 1 / 2 + 1 / n2p1
        b = 3 / 2
        c = 3 / 2 + 1 / n2p1
        z = -(t ** n2p1 / (4 * (n + 1) ** 2))
        return coeff * sc.special.hyp2f1(a, b, c, z)
    @np.vectorize
    def cf(t, n, a):
        coeff = a * t
        n2p1 = (2 * (n + 1))
        a = 1 / n2p1
        b = 1 / 2
        c = 1 + 1 / n2p1
        z = -(t ** n2p1 / (4 * (n + 1) ** 2))
        return coeff * sc.special.hyp2f1(a, b, c, z)
    # return as_complex(np.array([sf(t, exponent, norm), cf(t, exponent, norm)]))
    return sf(t, exponent, norm) + 1j * cf(t, exponent, norm)


def fresnel(t, exponent):
    return np.exp(1j * (np.abs(t) ** (exponent + 1) / (exponent + 1)))


def clothoid_gray(t, exponent=2, scale=1):
    """
    Gray (1997) defines a generalization of the Cornu spiral given by parametric equations:
    http://mathworld.wolfram.com/CornuSpiral.html
    http://books.google.fi/books?id=-LRumtTimYgC&lpg=PA49&hl=fi&pg=PA64#v=onepage&q&f=false

    In the equations below:
    exponent = n
    scale = a

    The arc length, curvature, and tangential angle of this curve are:
    s(t)     = at
    kappa(t) = -(t ** n) / a
    phi(t)	 = -(t ** (n + 1)) / (n + 1)

    The Cesàro equation is

    kappa = -(s ** n) / (a ** (n + 1))

    Note! This works great for generating clothoid spline curve segments.
    """
    return scale * np.cumsum(fresnel(t, exponent))


def clothoid_scaled(k, phi, s, start=0):
    points = 1000

    n, a, t = nat(k, phi, s)
    end = clothoid_gray(np.linspace(0, t, points), n, a/points)[-1]

    scale = 1 / np.abs(end)
    s *= scale; k /= scale

    n, a, t = nat(k, phi, s)
    return clothoid_gray(np.linspace(start * t, t, points), n, a/points)


def clothoid_gray_negative(t, exponent=2, scale=1):
    """
    Rotate 0..n values to handle negative values and fractional exponentation.
    """
    if (t < 0).any():
        raise ValueError("Only positive values for t accepted!")
    s = clothoid_gray(t, exponent, scale)
    return np.append(s[1:][::-1] * 1j * 1j, s)


def clothoid_params(n, a, t):
    s = a * t
    k = -(t ** n / a)
    phi = -(t ** (n + 1) / (n + 1))
    print "Arc length:\t%s\n" \
      "Curvature:\t%s\n" \
      "Tang. angle:\t%s\n" \
      "Tau angle:\t%s" % (s, k, phi, phi / (np.pi * 2))
    return (s, k, phi)


def plot_unit(scale = 3):
    o = Osc.from_ratio(1, 8000)
    plt.interactive(True)
    plt.plot(o[::].real, o[::].imag)
    plt.axis('equal')
    plt.axis((-scale, scale, -scale, scale))
    plt.show()


def n_from_phi_t(phi, t, branch=0):
    """
    Calculate n parameter from tangential angle (phi) and parameter t.

    About Lambert W function:
    http://en.wikipedia.org/wiki/Lambert_W_function
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.special.lambertw.html
    """
    return -1 - lambertw(np.log(t) / phi) / np.log(t)


def n_from_k_s_t(k, s, t):
    return np.log(-k * s / t) / np.log(t)


def n_a_phi(k, s, t):
    return (
        np.log(-k * s / t) / np.log(t),
        s / t,
        t ** (1 + np.log(-k * s / t) / np.log(t)) * np.log(t) / (-np.log(t) - np.log(-k * s / t)),
        )


def nat_pos(k, phi, s):
    """
    Get exponent (n), scaling (a) and parameter (t) from
    curvature (k), tangential angle (phi) and arc length (s).
    """
    ks = k * s
    b = -ks ** phi
    e = 1 / ks
    t = b ** e
    u = b ** -e
    return (
        np.log(-ks * u) / np.log(t),
        s * u,
        t
        )


def nat(k, phi, s):
    """
    Get exponent (n), scaling (a) and parameter (t) from
    curvature (k), tangential angle (phi) and arc length (s).
    """
    b = abspowersign(-k * s, phi)
    e = 1 / (k * s)
    return (
        np.log(-k * s * abspowersign(b, -e)) / np.log(abspowersign(b, e)),
        s * abspowersign(b, -e),
        abspowersign(b, e)
        )


def snk(a, t, phi):
    """
    Get arc length (s), exponent (n) and curvature (k) from
    scaling (a), parameter (t) and tangential angle (phi).
    """
    log = np.log
    exp = np.exp
    return (
        a * t,
        (-phi * lambertw(log(t ** (1 / phi))) + log(t ** (-phi))) / (phi * log(t)),
        -exp(-lambertw(log(t ** (1 / phi))) + log(t ** (-phi)) / phi) / a
        )


def nas(k, phi, t):
    log = np.log
    exp = np.exp
    return (
        -exp(-lambertw(log(t) / phi)) / k,
        log(exp(-lambertw(log(t) / phi)) / t) / log(t),
        -exp(-lambertw(log(t) / phi)) / (k * t)
        )


def clothoid_exp(t, exponent=2):
    """Function (1) at Rational Approximations for the Fresnel Integrals
    By Mark A. Heald -- http://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777277-6/S0025-5718-1985-0777277-6.pdf

    Useful values for t parameter seem to be between -pi and pi."""
    @np.vectorize
    def fresnel(t, exponent):
        return np.exp(1j * 0.5 * np.pi * (t**exponent))
    return np.cumsum(fresnel(t, exponent))


def clothoid_math(t, n=1, scale=1):
    """
    The clothoid (or Euler) spiral curve.
    Calculated using the Fresnel integrals.

    See: http://en.wikipedia.org/wiki/Euler_spiral
    """
    t = np.atleast_1d(t * np.pi * 2)
    b = np.sqrt(np.pi)
    s, c = sc.special.fresnel((t ** n) / b)
    return as_complex(scale * b * np.array([c, s]))


def clothoid(points, b=1, scaled=True):
    """
    The clothoid (or Euler) spiral curve.
    Calculated using the Fresnel integrals.

    See: http://en.wikipedia.org/wiki/Euler_spiral
    """
    points = np.atleast_1d(points)
    if scaled:
        k = np.sqrt(2.0 / np.pi)
        s, c = sc.special.fresnel(k * points / b) / k
    else:
        s, c = sc.special.fresnel(points / b)
    return as_complex(np.array([c, s]))


def clothoid_pow(points, limit=50):
    """
    The clothoid (or Euler) spiral curve.
    Calculated using the Fresnel integrals.

    See: http://en.wikipedia.org/wiki/Euler_spiral
    """
    points = np.atleast_1d(points)
    @np.vectorize
    def s(x):
        def st(t):
            return np.sin(t**2)
        return sc.integrate.quad(st, 0, x, epsrel=1e-6, epsabs=0, limit=limit)[0]
    @np.vectorize
    def c(x):
        def ct(t):
            return np.cos(t**2)
        return sc.integrate.quad(ct, 0, x, epsrel=1e-6, epsabs=0, limit=limit)[0]
    return as_complex(np.array([c(points), s(points)]))


def clothoid_slice(start, stop, n, endpoint=False):
    """
    A slice of clothoid spiral.
    """
    return clothoid(np.linspace(start, stop, n, endpoint))


def clothoid_windings(start, stop, n, endpoint=False):
    """
    A piece of clothoid with start and stop being winding values.
    """
    return clothoid(np.linspace(wind(start), wind(stop), n, endpoint), scaled=False)


def clothoid_length(turn, diff=0.5, n=1000, endpoint=True):
    """
    A piece of clothoid with starting point at turn windings (positive or negative)
    and going there to diff windings point.
    """
    return clothoid_windings(turn, turn + diff, n, endpoint)


def wind(x):
    """
    Normalizes points on clothoid to have x number of turns (windings) wrt. to origo
    """
    return np.sign(x) * np.sqrt(np.abs(x) * 4)


def unwind(x):
    """
    Get the point on clothoid that has x number of turns.
    """
    return np.sign(x) * (x ** 2 / 4)


def clothoid_angle(s):
    """
    Find the tangent angle of the clothoid curve at s points.
    """
    s = np.atleast_1d(s)
    a = 1.0 / np.sqrt(2.0)
    return np.sign(s) * ((a * s) ** 2.0) * np.pi


def cl_piece(start, stop, n, endpoint=False, scale=1, norm=False, fn=clothoid_windings):
    """
    Takes a piece of clothoid and rotates and translates it to unit vector.
    """
    angle = clothoid_angle(wind(start))
    curve = fn(start, stop, n, endpoint)
    coeff = 1 / np.abs(curve[-1]) if norm else 1
    rotated = (curve - curve[0]) * rect(coeff, angle)
    return rotated * scale


def curvature(a, b, c):
    """
    Discrete curvature estimation.

    See section "2.6.1 Discrete curvature estimation" at:
    http://www.dgp.toronto.edu/~mccrae/mccraeMScthesis.pdf
    """
    if a == b == c:
        return np.inf
    if is_collinear(a, b, c):
        return 0
    return 1 / circumcircle_radius(a, b, c)


def estimate_curvature_circle(signal):
    return np.array([curvature(*points) for points in consecutive(signal, 3)])


def ellipse_curvature(pts):
    ell = Ellipse.from_conjugate_diameters(pts[:3])
    return ell.curvature(np.angle(pts[1] - midpoint(pts[0], pts[2])) / pi2)


def estimate_curvature(signal):
    return np.array([ellipse_curvature(points) for points in consecutive(signal, 3)])


# Circular arcs

def arc(points, s=1.0):
    """
    An arc is an arc.

    Parameters
    ==========
    points:
        An array of points at which to sample the arc.
    s:
        The curvature parameter.
    """
    points = np.atleast_1d(points)
    return (1.0 / s) * (np.exp(1j * np.pi * 2.0 * points) - 1.0)


# Line segments

def line(a, b, n=100):
    """
    Basic line between two points.
    """
    return np.linspace(a, b, n)


def linediff(a, d, n=100):
    """
    Line with a point and a vector.
    """
    return line(a, a + d, n)
