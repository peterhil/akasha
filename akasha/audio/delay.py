"""
Time delay module.
"""

from akasha.math.functions import fixnans


class Delay:
    """
    Delay (shift) a sound object in time.
    """

    def __init__(self, time, sound):
        self.delay = time
        self.sound = sound

    def at(self, t):
        """
        Sample delayed sound object at sample times (t).
        """
        return fixnans(self.sound.at(t - self.delay))
