"""
What's the frequency Kenneth?
"""

import operator

from fractions import Fraction
from numbers import Number, Real

import numpy as np

from akasha.audio.generators import PeriodicGenerator
from akasha.timing import sampler
from akasha.types.numeric import RealUnit
from akasha.utils.python import _super
from akasha.utils.decorators import memoized
from akasha.utils.log import logger
from akasha.math import cents_diff
from akasha.settings import config


def is_audible(frequency):
    """Is the frequency audible?"""
    audible_min = config.frequency.AUDIBLE_MIN
    audible_max = config.frequency.AUDIBLE_MAX

    return audible_min < frequency < audible_max


def octaves():
    """Audible octaves based on config.frequency.BASE"""
    frequencies = config.frequency.BASE * 2.0 ** np.arange(-10, 15)
    return np.fromiter(filter(is_audible, frequencies), dtype=np.float64)


class FrequencyRatioMixin:
    """Mixin to enable memoization of sound objects with Frequency
    through rational approximation.
    """

    # FIXME: Basically this should be replaced with RationalUnit or at least
    # inherit from it.

    _hz = 0.0

    @classmethod
    def from_ratio(
        cls, ratio, den=False, *args, **kwargs
    ):  # pylint: disable=W1113
        """
        New instance from fractional ratio.
        """
        if den:
            ratio = Fraction(ratio, den)
        return cls(Fraction(ratio) * sampler.rate, *args, **kwargs)

    @property
    def frequency(self):
        """
        Frequency getter.
        """
        return self._hz

    @frequency.setter
    def frequency(self, hz):
        """
        Frequency setter.
        """
        # Use Trellis or other Cells clone?
        self._hz = float(hz) if isinstance(self, Frequency) else Frequency(hz)

    @property
    def ratio(self):
        """The wrapped and antialised rational approximation of
        self's frequency. This should be used when sampling the signal.
        """
        ratio = self.antialias(self.to_ratio(self._hz))
        if not sampler.prevent_aliasing or sampler.negative_frequencies:
            return self.wrap(ratio)
        return ratio

    @property
    def hz(self):
        """Hz depends on original frequency's rational
        approximation and sampling rate.
        """
        return float(self.ratio * sampler.rate)

    @property
    def period(self):
        """
        Period when sampling. Denominator of the ratio.
        """
        return self.ratio.denominator

    @property
    def order(self):
        """
        Ordering of the samples. Numerator of the ratio.
        """
        return self.ratio.numerator

    @staticmethod
    @memoized
    def to_ratio(freq, limit=sampler.rate * 2):
        """Returns a rationally approximated ratio (a Fraction)
        corresponding to the frequency.
        """
        # TODO: Investigate the right limit, take beating tones into account!
        # TODO: Check whether memoizing this is of any value.
        ratio = Fraction.from_float(
            float(freq) / sampler.rate
        ).limit_denominator(limit)
        if ratio != 0:
            approx = sampler.rate * ratio
            deviation = cents_diff(freq, approx)
            if deviation > config.logging_limits.FREQUENCY_DEVIATION_CENTS:
                logger.warning(
                    "Frequency approx %f for ratio %s deviates from "
                    "%.3f by %.16f%% cents",
                    approx,
                    ratio,
                    freq,
                    deviation,
                )
        return ratio

    @staticmethod
    def antialias(ratio):
        """Prevent antialiasing by producing silence (Zero frequency),
        if ratio is over 1/2 (Nyquist Frequency) or it is negative
        according to sampler settings.
        """
        if sampler.prevent_aliasing:
            if abs(ratio) > Fraction(1, 2):
                return Fraction(0, 1)
            if ratio < 0 and not sampler.negative_frequencies:
                return Fraction(0, 1)
        return ratio

    @staticmethod
    def wrap(ratio):
        """Wrap ratio modulo one, so Frequency.ratio of 9/8
        becomes 1/8 in Osc. This helps with numeric accuracy.
        """
        return ratio % 1

    def __nonzero__(self):
        """Zero frequency should be considered False."""
        return self.ratio != 0

    def _cmp(op):  # pylint: disable=E0213
        """Generate comparison methods."""

        def comparison(self, other):
            # pylint: disable=C0111,E1102
            if isinstance(other, FrequencyRatioMixin):
                return op(self.ratio, other.ratio)
            if isinstance(other, Number):
                return op(float(self), float(other))
            return NotImplemented

        comparison.__name__ = '__' + op.__name__ + '__'
        comparison.__doc__ = op.__doc__

        return comparison

    __eq__ = _cmp(operator.eq)
    __ge__ = _cmp(operator.ge)
    __gt__ = _cmp(operator.gt)
    __le__ = _cmp(operator.le)
    __lt__ = _cmp(operator.lt)

    def __float__(self):
        return float(self.hz)

    def __int__(self):
        return int(self.hz)


class Frequency(FrequencyRatioMixin, RealUnit, PeriodicGenerator):
    """Frequency class"""

    def __init__(self, hz, unwrapped=False):
        _super(self).__init__()
        if not isinstance(hz, Real):
            raise TypeError("Argument 'hz' must be a real number.")
        # Original frequency, independent of sampling rate or optimizations
        self._hz = float(hz)
        self.unwrapped = unwrapped

    @property
    def _unit(self):
        return '_hz'

    @property
    def ratio(self):
        if self.unwrapped:
            return self.to_ratio(self._hz)
        return _super(self).ratio

    @staticmethod
    @memoized
    def angles(ratio):
        """Normalized frequency (Tau) angles for one full period
        at ratio.
        """
        # TODO: Could be optimized by using conjugate for half of the samples.
        zero = np.zeros(1, dtype=np.float64)
        if np.all(zero == ratio):
            return zero
        step = 1.0 / ratio.denominator
        return ratio.numerator * np.arange(0, 1, step, dtype=np.float64)

    @property
    def cycle(self):
        """Frequency cycle of one period"""
        return self.angles(self.ratio)

    def at(self, t):
        return self._hz * t

    def __repr__(self):
        return f'Frequency({self._hz})'

    def __str__(self):
        return f'<Frequency: {self._hz} hz>'

    def __hash__(self):
        """hash(self), takes into account any rounding done on
        Frequency's initialisation."""
        return hash(self.ratio)

    def __eq__(self, other):
        """a == b, takes into account any rounding done on
        Frequency's initialisation."""
        if isinstance(other, Real):
            return self.ratio == Frequency(float(other)).ratio
        return NotImplemented
