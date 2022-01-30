#!/usr/bin/env python
# -*- coding: utf-8 -*-

from akasha.curves.kepler_orbit import KeplerOrbit


planet_data = {
    'mercury': {
        'perihelion': 0.307499,  # AU
        'semimajor': 0.387098,  # AU
        'period': 0.240846,  # years
        'eccentricity': 0.205630,
    },
    'venus': {
        'perihelion': 0.718440,
        'semimajor': 0.723332,
        'period': 0.615198,
        'eccentricity': 0.006772,
    },
    'earth': {
        'perihelion': 0.98329,  # AU
        'semimajor': 1.000001018,  # AU
        'period': 1,  # years
        'eccentricity': 0.0167086,
    },
    'moon': {
        'perihelion': 0.002424257,
        'semimajor': 0.00257,
        'period': 0.0748013,
        'eccentricity': 0.0549,
    },
    'mars': {
        'perihelion': 1.382,
        'semimajor': 1.523679,
        'period': 1.88082,
        'eccentricity': 0.0934,
    },
    'ceres': {
        'perihelion': 2.5586835997,
        'semimajor': 2.7691651545,
        'period': 4.61,
        'eccentricity': 0.07600902910,
    },
    'jupiter': {
        'perihelion': 4.9501,
        'semimajor': 5.2044,
        'period': 11.862,
        'eccentricity': 0.0489,
    },
    'saturn': {
        'perihelion': 9.0412,
        'semimajor': 9.5826,
        'period': 29.4571,
        'eccentricity': 0.0565,
    },
    'uranus': {
        'perihelion': 18.33,
        'semimajor': 19.2184,
        'period': 84.0205,
        'eccentricity': 0.046381,
    },
    'neptune': {
        'perihelion': 29.81,
        'semimajor': 30.07,
        'period': 164.8,
        'eccentricity': 0.008678,
    },
    'pluto': {
        'perihelion': 29.658,
        'semimajor': 39.482,
        'period': 247.94,
        'eccentricity': 0.2488,
    },
}


planets = {
    name: KeplerOrbit(name=name, **planet)
    for name, planet in planet_data.items()
}
