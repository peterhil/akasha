#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# E1101: Module 'x' has no 'y' member
# pylint: disable=E1101

"""
Clothoid splines.
http://www.dgp.toronto.edu/~karan/papers/sbim2008mccrae.pdf
"""

from __future__ import division

import numpy as np
import scipy as sc

from akasha.curves import Ellipse
from akasha.funct.itertools import consecutive
from akasha.math.geometry import orient, turtle_turns, vectors, wrap_ends
from akasha.math.geometry.curvature import estimate_curvature
from akasha.utils.log import logger
from akasha.math import (
    abslogsign,
    abspowersign,
    as_complex,
    cartesian,
    distances,
    lambertw,
    map_array,
    overlap,
    pad,
    pi2,
    rect,
)


def clothoid_erf(t):
    """Euler spiral defined using the error function:
    http://en.wikipedia.org/wiki/Fresnel_integral#Properties"""
    erfi = sc.special.erfi(t * (1 + 1j) / np.sqrt(2))
    return np.sqrt(np.pi / 2) * ((1 - 1j) / 2) * erfi


def clothoid_gray_hg(t, exponent=2, norm=1):
    """
    Generalization by Gray (1997):
    http://mathworld.wolfram.com/CornuSpiral.html

    Defined by generalized hypergeometric functions.
    Curvature is t**n.

    TODO: Something is not right here...
    """

    @np.vectorize
    def sf(t, n, a):
        coeff = (a * t ** (n + 2)) / ((n + 1) * (n + 2))
        n2p1 = 2 * (n + 1)
        a = 1 / 2 + 1 / n2p1
        b = 3 / 2
        c = 3 / 2 + 1 / n2p1
        z = -(t ** n2p1 / (4 * (n + 1) ** 2))
        return coeff * sc.special.hyp2f1(a, b, c, z)

    @np.vectorize
    def cf(t, n, a):
        coeff = a * t
        n2p1 = 2 * (n + 1)
        a = 1 / n2p1
        b = 1 / 2
        c = 1 + 1 / n2p1
        z = -(t ** n2p1 / (4 * (n + 1) ** 2))
        return coeff * sc.special.hyp2f1(a, b, c, z)

    # return as_complex(np.array([
    #     sf(t, exponent, norm),
    #     cf(t, exponent, norm)
    # ]))
    return sf(t, exponent, norm) + 1j * cf(t, exponent, norm)


def fresnel(t, exponent):
    return np.exp(1j * (np.abs(t) ** (exponent + 1) / (exponent + 1)))


def clothoid_gray(t, exponent=2, scale=1):
    """
    Gray (1997) defines a generalization of the Cornu spiral given
    by parametric equations: http://mathworld.wolfram.com/CornuSpiral.html
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


def clothoid_scaled(k, phi, s, start=0, distance=1):
    # TODO Maybe control the number of points by multiplying with s,
    # but beware of huge values of s!
    points = 1000

    n, a, t = nat(k, phi, s)
    end = clothoid_gray(np.arange(start * t, t, 1 / points), n, a / points)[
        -1
    ]
    scale = distance / np.abs(end)
    # s *= scale; k /= scale
    a *= scale

    # n, a, t = nat(k, phi, s)
    return clothoid_gray(np.arange(start * t, t, 1 / points), n, a / points)


def clothoid_scaleto(k, phi, s, start=0, target=1 + 0j):
    # TODO Maybe control the number of points by multiplying with s,
    # but beware of huge values of s!
    points = 1000

    n, a, t = nat(k, phi, s)
    sig = clothoid_gray(np.linspace(start * t, t, points), n, a / points)
    start, end = sig[0], sig[-1]
    # del sig

    vector = (target - start) / (end - start)
    s *= np.abs(vector)
    k /= np.abs(vector)

    n, a, t = nat(k, phi, s)

    return (
        clothoid_gray(np.linspace(start * t, t, points), n, a / points)
        * vector
    )


def clothoid_gray_negative(t, exponent=2, scale=1):
    """Rotate 0..n values to handle negative values and fractional
    exponentation.
    """
    if (t < 0).any():
        raise ValueError("Only positive values for t accepted!")
    s = clothoid_gray(t, exponent, scale)

    return np.append(s[1:][::-1] * 1j * 1j, s)


def curvature(n, a, t):
    return -(abspowersign(t, n) / a)


def tangent_angle(n, t):
    return -(abspowersign(t, (n + 1)) / (n + 1))


def arclength(a, t):
    return a * t


def kphis(n, a, t):
    k = curvature(n, a, t)
    phi = tangent_angle(n, t)
    s = arclength(a, t)
    return (k, phi, s)


def n_from_phi_t(phi, t):
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
    exponent = 1 + np.log(-k * s / t) / np.log(t)
    return (
        np.log(-k * s / t) / np.log(t),
        s / t,
        t ** exponent * np.log(t) / (-np.log(t) - np.log(-k * s / t)),
    )


def nat_pos(k, phi, s):
    """
    Get exponent (n), scaling (a) and parameter (t) from
    curvature (k), tangential angle (phi) and arc length (s).
    """
    ks = k * s
    b = -(ks ** phi)
    e = 1 / ks
    t = b ** e
    u = b ** -e
    return (np.log(-ks * u) / np.log(t), s * u, t)


def nat(k, phi, s):
    """
    Get exponent (n), scaling (a) and parameter (t) from
    curvature (k), tangential angle (phi) and arc length (s).
    """
    b = abspowersign(-k * s, phi)
    e = 1 / (k * s)
    num = abslogsign(-k * s * abspowersign(b, -e))
    den = abslogsign(abspowersign(b, e))

    return (num / den, s * abspowersign(b, -e), abspowersign(b, e))


def t_for_curvature(n, a, k):
    return abspowersign(-a * k, 1 / n)


def t_from_phi_n(phi, n):
    return abspowersign(phi * (n + 1), 1 / (n + 1))


def tf(k, phi, s):
    return abspowersign(abspowersign(-k * s, phi), 1 / (k * s))


def clothoid_segment(k, k2, phi, phi2, s, s2):
    t = tf(k, phi, s)
    t2 = tf(k2, phi2, s2)

    n = abslogsign(k2 / k) / abslogsign(t2 / t)
    a = abspowersign(-t, n) / k if k != 0 else abspowersign(-t2, n) / k2

    n2 = abslogsign(k / k2) / abslogsign(t / t2)
    a2 = abspowersign(-t, n2) / k2 if k2 != 0 else abspowersign(-t2, n2) / k
    logger.debug("n1: %s, a1: %s", n, a)
    logger.debug("n2: %s, a2: %s", n2, a2)

    return ((n, a, t, t2), (n2, a2, t, t2))


def clothoid_tangents(points, ends=None):
    """
    Find suitable tangent angles for a set of points.

    Angles are found by halving the angle of "turns" on the "turtle path"
    formed by the points.

    Parameters:
    ===========
    ends = handle end point tangents separately, and add to output

    Example:
    >>> points = np.array([1, 0, 2j, 2+2j, 3+1j, 4+2j, 5+3j, 5+2j])
    >>> tangents = clothoid_tangents(points) / pi2
    array([-0.125 , -0.125 , -0.0625,  0.125 ,  0.    , -0.1875])
    """
    if 'closed' == ends:
        return clothoid_tangents(wrap_ends(points), ends=None)
    directions = np.angle(vectors(points))
    turns = turtle_turns(points)
    logger.debug("Turns:\n%r\nSigns:\n%r", turns, np.sign(turns))
    current = 0

    def mod_angle(angle):
        # TODO Current is not changed?
        return np.fmod(current + angle, np.pi)

    tangents = map_array(mod_angle, turns / 2) + directions[:-1]
    if 'open' == ends:
        return np.concatenate(([directions[0]], tangents, [directions[-1]]))
    return tangents


def clothoid_tangents_simple(points, turns=False):
    """Find suitable tangent angles by angles between previous and
    next point on path.
    """
    points = pad(points, value=points[0], count=2, index=0)
    points = pad(points, value=points[-1], count=2, index=-1)

    tangents = np.angle(points[2:] - points[:-2])[1:-1]

    if turns:
        # TODO check that this gives right results!
        directions = np.angle(vectors(points))
        return tangents + directions[2:-1]
    else:
        return tangents


def snk(a, t, phi):
    """
    Get arc length (s), exponent (n) and curvature (k) from
    scaling (a), parameter (t) and tangential angle (phi).
    """
    log = np.log
    exp = np.exp
    lambert = lambertw(log(t ** (1 / phi)))
    log_t_phi = log(t ** (-phi))
    return (
        a * t,
        (-phi * lambert + log_t_phi) / (phi * log(t)),
        -exp(-lambert + log_t_phi / phi) / a,
    )


def nas(k, phi, t):
    log = np.log
    exp = np.exp
    return (
        -exp(-lambertw(log(t) / phi)) / k,
        log(exp(-lambertw(log(t) / phi)) / t) / log(t),
        -exp(-lambertw(log(t) / phi)) / (k * t),
    )


def clothoid_exp(t, exponent=2):
    """Function (1) at Rational Approximations for the Fresnel Integrals
    By Mark A. Heald:
    http://www.ams.org/journals/mcom/1985-44-170/S0025-5718-1985-0777277-6/S0025-5718-1985-0777277-6.pdf

    Useful values for t parameter seem to be between -pi and pi."""

    @np.vectorize
    def fresnel(t, exponent):
        return np.exp(1j * 0.5 * np.pi * (t ** exponent))

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
    kwargs = dict(
        epsrel=1e-6,
        epsabs=0,
        limit=limit,
    )

    @np.vectorize
    def s(x):
        def st(t):
            return np.sin(t ** 2)

        return sc.integrate.quad(st, 0, x, **kwargs)[0]

    @np.vectorize
    def c(x):
        def ct(t):
            return np.cos(t ** 2)

        return sc.integrate.quad(ct, 0, x, **kwargs)[0]

    return as_complex(np.array([c(points), s(points)]))


def clothoid_slice(start, stop, n, endpoint=False):
    """A slice of clothoid spiral."""
    return clothoid(np.linspace(start, stop, n, endpoint))


def clothoid_windings(start, stop, n, endpoint=False):
    """A piece of clothoid with start and stop being winding values."""
    winds = np.linspace(wind(start), wind(stop), n, endpoint)
    return clothoid(winds, scaled=False)


def clothoid_length(turn, diff=0.5, n=1000, endpoint=True):
    """A piece of clothoid with starting point at turn windings
    (positive or negative) and going there to diff windings point.
    """
    return clothoid_windings(turn, turn + diff, n, endpoint)


def wind(x):
    """Normalizes points on clothoid to have x number of turns
    (windings) wrt. to origo
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


def cl_piece(
    start, stop, n, endpoint=False, scale=1, norm=False, fn=clothoid_windings
):
    """
    Takes a piece of clothoid and rotates and translates it to unit vector.
    """
    angle = clothoid_angle(wind(start))
    curve = fn(start, stop, n, endpoint)
    coeff = 1 / np.abs(curve[-1]) if norm else 1
    rotated = (curve - curve[0]) * rect(coeff, angle)
    return rotated * scale


def clothoid_arc_length(para):
    ell = Ellipse.from_conjugate_diameters(para[:3])
    arc_length = ell.arc_length(np.angle(para[:3][::-1] - ell.origin) / pi2)
    return np.ediff1d(arc_length[::-1])


def estimate_arc_length(signal, mean=sc.stats.hmean):
    arr = np.array(
        [clothoid_arc_length(points) for points in consecutive(signal, 3)]
    )
    fst, snd = np.append(arr, [[arr[-1, -1], arr[0, 0]]], axis=0).T
    snd = np.roll(snd, 1)
    # TODO try other means (pun intended) also, see:
    # https://en.wikipedia.org/wiki/Pythagorean_means
    #
    # Harmonic mean always gives the smallest value (for positive numbers),
    # then geometric, then arithmetic mean (average).
    #
    # Harmonic mean is the same as 1/np.mean(1/arr), but doesn't
    # make sense for negative values (results will be +/-inf).
    return mean(np.abs(np.array([fst, snd])), axis=0)


def clothoid_curve(n, a, t, t2, use_range=True):
    # s = np.abs(a) * np.abs(t2 - t)
    # resolution = 10  # dots per arc length unit
    # points = s * resolution
    points = 1000
    # if not np.isnan(s):
    #     points = np.min((500, points * s))
    # logger.debug("Clothoid curve() -- Arc length: %r" % s)
    logger.debug("Using %i points.", points)
    direction = -1 if np.sign(t2 - t) == -1 else 1
    step = direction * a / points

    if use_range:
        tt = np.arange(t, t2 + step, step)
    else:
        tt = np.linspace(t, t2, np.abs(t2 - t) * points + 1, endpoint=True)
    print(tt)
    return clothoid_gray(tt, n, a * step)


def clothoid_segments(signal, ends='open', mean=np.mean):
    directions = np.angle(vectors(signal))
    scales = 1 / distances(signal)

    # Points
    phi = clothoid_tangents(signal, ends)
    k = estimate_curvature(signal, ends)  # / np.append(scales, 1)

    # Segments
    # s = np.ones(len(signal) - 1)
    # s = estimate_arc_length(signal, mean) / scales
    # s = distances(signal)
    s = scales

    k[1:-1] = -np.sign(turtle_turns(signal)) * np.abs(k[1:-1])

    ks = overlap(k, 2).T
    phis = overlap(phi, 2).T
    # TODO find out correct lengths of the parts of clothoid segment!!!
    # ss = repeat(s / 2, 2).reshape(2, len(s)).T
    ss = np.concatenate((s / 2, s)).reshape(2, len(s)).T

    segments = np.hstack((ks, phis, ss))
    logger.debug("Segments (k, k2, phi, phi2, s, s2):\n%r", segments)

    params = np.apply_along_axis(lambda p: clothoid_segment(*p), 1, segments)
    logger.debug("Params (n, a, t, t2):\n%r", params)

    def add_dim(arr):
        return np.resize(arr, (len(arr), 1))

    def straighten(s):
        return s / rect(1, np.angle(s[-1] - s[0]))

    # clothoids = map(
    #     lambda n, a, t, t2: orient(clothoid_curve(n, a, t, t2)),
    #     *params.T
    # )
    # Works only if clothoid_curve always gives the same number of points!
    clothoids = np.apply_along_axis(
        lambda p: orient(clothoid_curve(*p)), 1, params
    )
    path = cartesian(distances(signal), directions)
    clothoids = add_dim(path) * clothoids + add_dim(signal[:-1])

    return (clothoids, segments, params)


def ellipse(t):
    e = Ellipse.from_conjugate_diameters(t[:3])
    # a, b = (np.angle([t[0], t[2]] - e.origin) - e.angle) / pi2
    # if np.sign(turns(t)) <= 0:
    #     a -= 0.5; b += 0.5
    # print(a, b, np.sign(turns(t)))
    a, b = 0, 0.96
    return e.at(np.linspace(a, b, 200))


def ellipses(signal):
    return np.array(
        [ellipse(points) for points in consecutive(signal, 3)]
    ).flatten()


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
