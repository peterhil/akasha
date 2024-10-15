"""
Releasable mixin module
"""

import numpy as np

from akasha.audio.mixins.composite import Composite


class Releasable(Composite):
    """Mixin to create playable composite sound object that can be
    released on key up.
    """

    def _releasable_components(self):
        """
        Return components which can be released on key up.
        """
        return self._components_with_attribute('release_at')

    def release_at(self, time=None):
        """
        Set release time for components.
        """
        if not np.isreal(time):
            raise ValueError("Release time should be a real number!")

        return [c.release_at(time) for c in self._releasable_components()]
