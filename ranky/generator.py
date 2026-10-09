#########################
#### JUDGE GENERATOR ####
#########################

# Generate synthetic judges (ballots) by adding noise to a reference ranking.

import numpy as np
import random

class Generator():
    """ Base generator: sample judges from a reference ranking.

    Subclasses define how a judge is sampled by overriding `_sample`. The base
    class returns the reference ranking unchanged.

    For example, to sample 10 judges from a ranking of 5 candidates:

        g = rk.SwapGenerator()
        g.fit([5, 4, 3, 2, 1])
        m = g.sample(n=10, N=2)  # shape (5, 10)
    """
    def __init__(self):
        self.r = None

    def fit(self, r):
        """ Store the reference ranking.

        Args:
            r: Reference ranking (1D array-like), or an integer n to use range(n).
        """
        if isinstance(r, int):
            r = list(range(r))
        if len(r) < 2:
            raise ValueError('The reference ranking must contain at least two candidates.')
        self.r = r
        return self

    def sample(self, n=1, return_single=True, **kwargs):
        """ Sample judges.

        Args:
            n: Number of judges to draw.
            return_single: If True, return a 1D array when n = 1.
            **kwargs: Arguments passed to the sampling function (see subclasses).

        Returns:
            A matrix with one row per candidate and one column per judge.
        """
        if self.r is None:
            raise RuntimeError('The generator must be fitted before sampling.')
        if n == 1 and return_single:
            return np.array(self._sample(**kwargs)) # return one array
        return np.array([self._sample(**kwargs) for _ in range(n)]).T # return a matrix

    def _sample(self):
        """ Sample one judge. To be overridden by subclasses.
        """
        return self.r

class SwapGenerator(Generator):
    """ Generate judges by swapping neighbors in the reference ranking. """
    def _sample(self, N=1, p=1):
        """ Sample one judge by swapping random neighbors N times with probability p.

        Args:
            N: Number of swaps.
            p: Probability (in ]0;1]) of performing each swap.
        """
        return neighbors_swap(self.r, N=N, p=p)

class GaussianGenerator(Generator):
    """ Generate judges by adding Gaussian noise to the reference ranking. """
    def _sample(self, loc=0, scale=1):
        """ Sample one judge by adding Gaussian noise to the reference ranking.

        Args:
            loc: Mean of the noise.
            scale: Standard deviation of the noise.
        """
        return gaussian_noise(self.r, loc=loc, scale=scale)

################
###  NOISES  ###
################

def neighbors_swap(r, N=1, p=1):
    """ Swap random neighbors N times with probability p.

    Args:
        r: The ranking to disturb. It is copied, not modified in place.
        N: Number of iterations.
        p: Probability (in ]0;1]) of disturbing on each iteration.
    """
    if (p <= 0) or (p > 1):
        raise ValueError('p must be in ]0;1]')
    _r = r.copy()
    for _ in range(N):
        i1 = np.random.randint(len(r) - 1) # uniform selection
        i2 = i1 + 1
        if np.random.random() < p: # probability
            _r[i1], _r[i2] = _r[i2], _r[i1] # swap neighbors
    return _r

def ranking_noise(r, method='swap', n=1, p=1):
    """ Swap or tie random pairs of candidates.

    Args:
        r: The ranking to disturb. It is copied, not modified in place.
        method: 'swap' to swap two values, 'tie' to give the first one the value of the second one.
        n: Number of iterations.
        p: Probability (in ]0;1]) of disturbing on each iteration.
    """
    if (p <= 0) or (p > 1):
        raise ValueError('p must be in ]0;1]')
    if method not in ['swap', 'tie']:
        raise ValueError('Unknown ranking noise method: {}.'.format(method))
    _r = r.copy()
    for _ in range(n):
        i1, i2 = random.sample(range(len(r)), 2)
        if np.random.random() < p: # probability
            if method == 'swap':
                _r[i1], _r[i2] = _r[i2], _r[i1]
            elif method == 'tie':
                _r[i1] = _r[i2]
    return _r

def gaussian_noise(r, loc=0, scale=1):
    """ Add Gaussian noise to r.

    Args:
        r: 1D array-like.
        loc: Mean of the noise.
        scale: Standard deviation of the noise.
    """
    r = np.asarray(r)
    noise = np.random.normal(loc, scale, r.shape)
    return r + noise
