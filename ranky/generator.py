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
        self.r = r

    def sample(self, n=1, loc=0, scale=1):
        """ Sample judge according to the function.

        Args:
            n: number of samples to draw.
        """
        #m = [ranking_noise(self.r) for _ in range(n)]
        m = [gaussian_noise(self.r, loc=loc, scale=scale) for _ in range(n)]
        return np.array(m).T

### NOISES ###

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

### FUNCTIONS ###

def identity(r):
    return r
