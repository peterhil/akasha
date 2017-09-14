#!/usr/bin/env python
# encoding: utf-8

import numpy as np


class Composite(object):
    """
    Mixin to create composite sound object from components.
    """

    components = []

    def _components_with_attribute(self, attribute):
        """
        Return components which have the named attribute.
        """
        return np.array([component for component in self.components if hasattr(component, attribute)])
