"""
Composite sound object mixin.
"""

import numpy as np


class Composite:
    """Mixin to create composite sound object from components."""

    components = []

    def _components_with_attribute(self, attribute):
        """Return components which have the named attribute."""
        return np.array([c for c in self.components if hasattr(c, attribute)])
