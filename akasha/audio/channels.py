"""
Audio channels module
"""

import numpy as np


def num_channels(data):
    """
    Number of audio channels in data.
    """
    return 1 if np.ndim(data) <= 1 else data.shape[1]
