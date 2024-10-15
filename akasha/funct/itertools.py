"""
Functional programming utility functions.

Mostly from itertools recipes:
http://docs.python.org/library/itertools.html#recipes
"""

import collections
import numpy as np

from itertools import count, islice, tee

from akasha.utils.log import logger


def take(n, iterable):
    """
    Take first n items of the iterable. Return them as a Numpy array.
    """
    return np.fromiter(islice(iterable, n), dtype=iterable.dtype)


def pairwise(iterable):
    """
    Pair consecutive elements.

    pairwise([s0, s1, s2, s3, ...]) -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def consume(iterator, n=None):
    """
    Advance the iterator n-steps ahead. If n is none, consume entirely.
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


# Other iterators and generators


def count_step(start=0, step=1):
    """
    Get an iterator that advances from start with step at a time.
    """
    g = (start + step * i for i in count())
    yield next(g)
    while True:
        yield next(g)


def consecutive(signal, n):
    """
    Iterates over n consecutive samples from the signal at a time.

    Example
    -------

    >>> list(consecutive(np.linspace(0, 1, 5), 2))
    [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    """
    return zip(*[signal[start:] for start in range(n)])


def blockwise(iterable, step=1, start=0):
    """
    Blockwise iterator with blocksize equal to step parameter.
    """
    if np.sign(step) == -1:
        logger.warning(
            "Blockwise will lose first value on negative step, "
            "because of Numpy array's indexing bug."
        )

    def reset(start=start):
        """
        Reset the blockwise iterator.
        """
        return pairwise(count_step(start, step))

    blocks = reset()
    while True:
        try:
            indices = next(blocks)
            block = iterable[slice(*np.append(indices, np.sign(step)))]
        except StopIteration:
            break
        except GeneratorExit:
            logger.debug("Blockwise generator exit.")
            break

        if len(block) > 0:
            val = yield block  # sending values into generator with send
            while val is not None:
                if val == 'current':
                    val = yield indices
                if val == 'reset':
                    blocks = reset()
                    indices = next(blocks)
                    sl = slice(*np.append(indices, np.sign(step)))
                    block = iterable[sl]
                    val = yield block
        else:
            raise StopIteration

    del block, blocks, indices
