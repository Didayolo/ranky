#########################
#### JUDGE GENERATOR ####
#########################

import numpy as np
import random
import ranky.ranking as rk

class Generator():
    def __init__(self):
        """ Create a Generator object.
        """
        self.r = None

    def fit(self, r):
        """ Store the reference ranking.
        """
        if isinstance(r, int):
            self.r = list(range(r))
        else:
            if len(r) < 2:
                raise Exception('The reference ranking must contains at least two candidates.')
            self.r = r

    def sample(self, n=1, return_single=True, **kwargs):
        """ Sample judge according to the function.

        Args:
            n: number of samples to draw.
            return_single: if True, return a simple 1D array when n = 1.
        """
        if self.r is None:
            raise Exception('The generator must be fitted before sampling.')
        if n == 1 and return_single:
            return np.array(self._sample(**kwargs)) # return one array
        return np.array([self._sample(**kwargs) for _ in range(n)]).T # return a matrix

    def _sample(self):
        """ Sampling function to be re-written when inheriting this class.
        """
        return self.r

class SwapGenerator(Generator):
    def _sample(self, n=1, N=1, p=1):
        """ Sample n judges by swaping neighbors N times with probability p.

        Args:
            n: number of samples to draw.
            N: number swapings.
            p: probability (in ]0;1]) of disturbing on each iteration.
        """
        return neighbors_swap(self.r, N=N, p=p)

class GaussianGenerator(Generator):
    def _sample(self, n=1, loc=0, scale=1):
        """ Sample n judges by normally disturb the original ranking.

        Args:
            n: number of samples to draw.
        """
        return gaussian_noise(self.r, loc=loc, scale=scale)

################
###  NOISES  ###
################

def neighbors_swap(r, N=1, p=1):
    """ Swap random neighbors n times with probability p.

    Args:
        r: the ranking to disturb.
        n: number of iterations.
        p: probability (in ]0;1]) of disturbing on each iteration.
    """
    if (p <= 0) or (p > 1):
        raise('p must be in ]0;1]')
    _r = r.copy()
    for _ in range(N):
        i1 = np.random.randint(len(r) - 1) # uniform selection
        i2 = i1 + 1
        if np.random.randint(1/p) == 0: # probability
            _r[i1], _r[i2] = _r[i2], _r[i1] # swap neighbors
    return _r

def ranking_noise(r, method='swap', n=1, p=1):
    """ Ranking noise.

    Args:
        r: the ranking to disturb.
        method: 'swap' or 'tie'.
        n: number of iterations.
        p: probability of disturbing on each iteration in ]0;1].
    """
    if (p <= 0) or (p > 1):
        raise('p must be in ]0;1]')
    _r = r.copy()
    for _ in range(n):
        #i1, i2 = np.random.randint(len(r)), np.random.randint(len(r)) # WARNING: sometimes i1 == i2
        i1, i2 = random.sample(range(len(r)), 2)
        if np.random.randint(1/p) == 0: # probability
            if method == 'swap':
                _r[i1], _r[i2] = _r[i2], _r[i1]
            elif method == 'tie':
                _r[i1] = _r[i2]
            else:
                raise('Unknown ranking noise method: {}.'.format(method))
    return _r

def gaussian_noise(r, loc=0, scale=1):
    if not isinstance(r, np.ndarray):
        r = np.array(r)
    noise = np.random.normal(loc, scale, r.shape)
    return r + noise
