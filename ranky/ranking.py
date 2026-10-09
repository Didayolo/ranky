#######################
### RANKING SYSTEMS ###
#######################

# Conventions: a preference matrix has candidates as rows and judges as columns.
# Most ranking systems take `axis`, the axis of the judges (1 by default).
# Some lower-level functions (`rank`, `evolution_strategy`, `brute_force`,
# `consensus`) take the axis of the candidates instead; see their docstrings.

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import differential_evolution
from random import random as _random
from tqdm import tqdm
import itertools as it
import ranky as rk
from .metric import centrality

#####################
##### Functions #####
#####################

# Convert to ranking
def rank(m, axis=0, method='average', ascending=False, reverse=False):
    """ Replace values by their rank along an axis.

    By default, higher is better: the highest value gets rank 1.

    Args:
        m: Score matrix or 1D array of scores.
        axis: Candidates axis.
        method: How to rank ties: 'average', 'min', 'max', 'dense' or 'ordinal'.
            See `scipy.stats.rankdata`.
        ascending: If True, lower is better.
        reverse: If True, reverse the order. Setting both `ascending` and
            `reverse` to True gives back the default order.

    Returns:
        Ranks with the same shape and type as `m` (np.ndarray, pd.Series or pd.DataFrame).

    >>> rank([0.2, 0.9, 0.5])
    array([3., 1., 2.])
    """
    if isinstance(m, list):
        m = np.array(m)
    if ascending == reverse: # greater is better (descending order)
        m = -m # take the opposite to inverse rank
    r = np.apply_along_axis(rankdata, axis, m, method=method) # convert values to ranking in all rows or columns
    return process_vote(m, r)

def weigher(r, method='hyperbolic'):
    """ Map a rank to a weight.

    Ranks start at zero (most important element). The 'hyperbolic' method maps
    rank r to the weight 1 / (r + 1).

    Args:
        r: Nonnegative integer (or array of integers) to weight.
        method: Weighing method. Only 'hyperbolic' is available.
    """
    if method == 'hyperbolic':
        return 1 / (r + 1)
    else:
        raise ValueError('Unknown method: {}'.format(method))

def contains_ties(r):
    """ Return True if r contains at least two equal values.

    Args:
        r: 1D array-like of scores or ranks.
    """
    s = np.sort(np.asarray(r))
    return bool(np.any(s[1:] == s[:-1]))

# remove rows (candidates) or columns (voters)
def bootstrap(m, axis=0, n=None, replace=True, return_holdout=False):
    """ Sample along an axis, with replacement by default.

    By convention rows represent candidates and columns represent judges.

    Args:
        m: Matrix to sample from (np.ndarray or pd.DataFrame).
        axis: Axis to sample.
        n: Number of examples to sample. By default it is the size of the matrix along the axis.
        replace: Sample with or without replacement. It is not bootstrap if the sampling is done without replacement.
        return_holdout: If True, return a tuple (bootstrap, out-of-bag set).
    """
    if n is None:
        n = m.shape[axis]
    idx = np.random.choice(m.shape[axis], n, replace=replace)
    bootstrap = np.take(m, idx, axis=axis)
    if return_holdout:
        holdout_idx = np.setdiff1d(np.arange(m.shape[axis]), idx)
        holdout = np.take(m, holdout_idx, axis=axis)
        return bootstrap, holdout
    else:
        return bootstrap

def joint_bootstrap(m_list, axis=0, n=None, replace=True):
    """ Apply the same bootstrap to all matrices in m_list.

    All matrices must have the same size along `axis`.

    Args:
        m_list: List of matrices.
        axis: Axis to sample.
        n: Number of examples to sample. By default it is the size of the matrices along the axis.
        replace: Sample with or without replacement.
    """
    size = m_list[0].shape[axis]
    if n is None:
        n = size
    idx = np.random.choice(size, n, replace=replace)
    return [np.take(m, idx, axis=axis) for m in m_list]

def top_k_method(D, F, k=1, reverse=False):
    """ Select a winner from two leaderboards D and F (development and final).

    Only the k best candidates of D access the final phase; the winner is the
    best of them in F.

    Args:
        D: 1D array-like of scores, representing the development phase (or public leaderboard).
        F: 1D array-like of scores, representing the final phase (or private leaderboard).
        k: Number of candidates that access the final phase.
        reverse: If True, lower is better (by default higher is better).

    Returns:
        The index (or label, for pd.Series) of the winner.
    """
    D, F = to_series(D), to_series(F)
    top_k = select_k_best(D, k=k, reverse=reverse)
    return select_best(F[top_k], reverse=reverse)

def select_k_best(m, k=1, reverse=False):
    """ Return the indices (or labels) of the k best candidates in a 1D array.

    Args:
        m: 1D array-like of scores.
        k: Number of best candidates to return.
        reverse: If True, lower is better (by default higher is better).
    """
    m = to_series(m)
    if k < 1 or k > len(m):
        raise ValueError('k must be between 1 and {}, got {}.'.format(len(m), k))
    return m.sort_values(ascending=reverse).index[:k]

def select_best(m, reverse=False):
    """ Return the index (or label) of the best candidate in a 1D array.

    Args:
        m: 1D array-like of scores.
        reverse: If True, lower is better (by default higher is better).
    """
    return select_k_best(m, k=1, reverse=reverse)[0]

def take_by_axis(m, indices, axis=0):
    """ Return a new array with only the rows (axis=0) or columns (axis=1) given by `indices`.
    """
    m = np.array(m)
    if axis == 0:
        return m[indices, :]
    elif axis == 1:
        return m[:, indices]
    else:
        raise ValueError("axis must be 0 or 1 for a 2D array")

def is_series(m):
    """ Return True if m is a pd.Series. """
    return isinstance(m, pd.Series)

def is_dataframe(m):
    """ Return True if m is a pd.DataFrame. """
    return isinstance(m, pd.DataFrame)

def to_series(m):
    """ Convert a 1D array-like (or a single column matrix) to a pd.Series. """
    if is_dataframe(m) and m.shape[1] == 1:
        m = m.iloc[:, 0]
    elif isinstance(m, np.ndarray) and m.ndim == 2 and m.shape[1] == 1: # "column array"
        m = m.reshape(m.shape[0])
    if not is_series(m): # cast to pd.Series if needed
        m = pd.Series(m)
    return m

def process_vote(m, r, axis=1):
    """ Give the result `r` the index/column names of the input `m`, if any.

    Args:
        m: Original matrix of scores (pd.DataFrame, pd.Series or np.ndarray).
        r: The result (array-like).
        axis: Axis of the judges. If r is 1D, it is indexed by the other axis of m.

    Returns:
        r as a pd.Series or pd.DataFrame if m is a pandas object, unchanged otherwise.
    """
    if is_dataframe(m):
        if len(r.shape) == 1: # Series
            if axis == 0: # Voting axis
                r = pd.Series(r, m.columns) # Participants names
            elif axis == 1:
                r = pd.Series(r, m.index)
        elif len(r.shape) == 2: # DataFrame
            r = pd.DataFrame(r, index=m.index, columns=m.columns)
    elif is_series(m): # From Series to Series
        r = pd.Series(r, m.index)
    return r

#################################
####### RANKING SYSTEMS #########
#################################

# All ranking systems below return one value per candidate. Depending on the
# method, higher is better (scores, number of wins) or lower is better (ranks).

#################################
##### 1. CLASSICAL METHODS #######
#################################

def dictator(m, axis=1):
    """ Random dictator: return the ballot of a random judge.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
    """
    voter = np.random.randint(m.shape[axis]) # select a column number
    r = np.take(np.array(m), voter, axis=axis)
    return process_vote(m, r, axis=axis)

def average_rank(m, axis=1, method='mean', reverse=False):
    """ Average rank (Borda count).

    Each judge's scores are converted to ranks (1 is best), then the ranks are
    averaged. Lower is better in the output.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
        method: 'mean' or 'median'.
        reverse: If True, lower scores are better.
    """
    ranking = rank(m, axis=1-axis, reverse=reverse)
    ranking = pd.DataFrame(np.asarray(ranking))
    if method == 'mean':
        r = ranking.mean(axis=axis)
    elif method == 'median':
        r = ranking.median(axis=axis)
    else:
        raise ValueError('Unknown method for average rank system: {}'.format(method))
    return process_vote(m, np.asarray(r), axis=axis)

def borda(m, axis=1, method='mean', reverse=False):
    """ Alias of `average_rank`.
    """
    return average_rank(m, axis=axis, method=method, reverse=reverse)

def majority(m, axis=1):
    """ Majority judgement: median score of each candidate.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
    """
    r = np.median(m, axis=axis)
    return process_vote(m, r, axis=axis)

def score(m, axis=1):
    """ Score voting (range voting): mean score of each candidate.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
    """
    r = np.mean(m, axis=axis)
    return process_vote(m, r, axis=axis)

def uninominal(m, axis=1, turns=1, keep_ranking=True):
    """ Uninominal voting (multi-turn instant-runoff).

    Each judge votes for their favorite candidate. With turns >= 2, the `turns`
    best candidates go to the next turn, and so on.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
        turns: Number of turns. Must be lower than the number of candidates.
        keep_ranking: If False, candidates eliminated in the first turn get a score of 0.

    Returns:
        Number of votes of each candidate (in the last turn they reached).
    """
    _m = m
    m = np.array(m)
    if turns >= m.shape[1-axis]: # if more turns than candidates
        raise ValueError('The number of turns must be lower than the number of candidates.')
    ranking = rank(m, axis=1-axis) # convert to rank
    r = (ranking == 1).sum(axis=axis)  # count number of uninominal vote (first in judgement)
    if turns >= 2:
        bests = np.argsort(r)[-turns:] # take the `turns` highest scores
        m2 = take_by_axis(m, bests, axis=1-axis) # take the best candidates
        r2 = uninominal(m2, axis=axis, turns=turns-1) # recursive call
        # re-create a general ranking with the results of the last turn
        if not keep_ranking:
            r = np.zeros_like(r)
        r[bests] = r2
    return process_vote(_m, r, axis=axis)

def pairwise(m, axis=1, wins=None, return_graph=False, score=False, **kwargs):
    """ Pairwise method.

    Compute the result of all one-to-one matches between candidates. The result
    of a match is given by the `wins` function, and each candidate's final score
    is the sum of its results.

    Args:
        m: 2D matrix of scores (preference matrix).
        axis: Axis of judges.
        wins: Function wins(a, b) returning the score of a against b.
            `rk.copeland_wins` by default. See the `rk.duel` module for other options.
        return_graph: If True, return a tuple (scores, graph), where graph[i, j]
            is the result of candidate i against candidate j.
        score: If True, divide the results by (n - 1), with n the number of
            candidates, to get values between 0 and 1.
        **kwargs: Arguments passed to the `wins` function.
    """
    if wins is None:
        wins = rk.copeland_wins
    _m = m
    m = np.array(m)
    n = m.shape[1-axis]
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: # no comparison with itself
                c1, c2 = np.take(m, i, 1-axis), np.take(m, j, 1-axis)
                graph[i][j] = wins(c1, c2, **kwargs)
    r = np.sum(graph, axis=1) # collect candidates score against all opponents
    r = process_vote(_m, r, axis=axis)
    if score:
        r = r / (n - 1)
    if return_graph:
        return r, graph
    return r

def copeland(m, axis=1, **kwargs):
    """ Copeland's method.

    Alias of `rk.pairwise` with `rk.copeland_wins` as the wins function.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
        **kwargs: Arguments passed to `rk.pairwise`.
    """
    return pairwise(m, axis=axis, wins=rk.copeland_wins, **kwargs)

def kemeny_young(m, axis=1, **kwargs):
    """ Kemeny-Young method.

    Alias of `rk.center` with Kendall tau as the metric: the result is the
    ranking closest to all judges according to Kendall's distance.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
        **kwargs: Arguments passed to `rk.center`.
    """
    return center(m, axis=axis, method='kendalltau', **kwargs)


#################################
### 2. COMPUTATIONAL METHODS ####
#################################

# Based on optimization.

def brute_force(m, axis=0, method='swap'):
    """ Find the most central ranking by trying all permutations.

    Only usable with a small number of candidates (n! rankings are evaluated).
    Ties are not considered. If several rankings are optimal, the first one found
    is returned.

    Args:
        m: 2D matrix of scores.
        axis: Axis of candidates.
        method: Metric used to compute the centrality (see `rk.centrality`).

    Returns:
        A tuple giving the position of each candidate (0 is the worst).
    """
    best_score = -np.inf
    best_r = None
    for r in tqdm(list(it.permutations(range(m.shape[axis])))): # all possible rankings
        score = centrality(m, r, axis=axis, method=method)
        if best_r is None or score > best_score:
            best_score = score
            best_r = r
    return best_r

def random_swap(r, n=1, tie=0.1):
    """ Randomly swap two values in r (or tie them).

    Used as the mutation operator by `evolution_strategy`.

    Args:
        r: Ranking (np.ndarray) to modify. It is copied, not modified in place.
        n: Number of consecutive changes.
        tie: Probability of creating a tie instead of a swap.
    """
    _r = r.copy()
    for _ in range(n):
        i1, i2 = np.random.randint(len(r)), np.random.randint(len(r))
        if np.random.random() < tie: # generate tie with some probability
            _r[i1] = _r[i2]
        else: # swap two values
            _r[i1], _r[i2] = _r[i2], _r[i1]
    return _r

def evolution_strategy(m, axis=0, mu=10, l=2, epochs=50, n=1, tie=0.1, method='swap', history=False, verbose=False):
    """ Search the most central ranking with a (mu + mu*l) evolution strategy.

    Args:
        m: 2D matrix of scores.
        axis: Axis of candidates.
        mu: Population size.
        l: Each generation produces mu * l offspring.
        epochs: Number of generations.
        n: Number of swaps performed during a single mutation.
        tie: Probability of creating a tie instead of a swap during mutation.
        method: Metric used to compute the centrality (see `rk.centrality`).
        history: If True, return a tuple (ranking, best score of each generation).
        verbose: If True, plot the learning curve and print the best score.
    """
    r = np.arange(m.shape[axis])
    h = []
    population = [sorted(r, key=lambda k: _random()) for _ in range(mu)] # mu random ranked ballots
    best_ranking = population[0] # initialize best_ranking
    for epoch in tqdm(range(epochs)):
        offspring = [random_swap(x, n=n, tie=tie) for x in population*l] # random swaps to generate new ranked ballots
        offspring.append(best_ranking) # keep the previous best in case no child beats it
        scores = [centrality(m, child, axis=axis, method=method) for child in offspring] # compute fit function
        idx_best = np.argsort(scores)[len(scores)-mu:]
        population = list(np.array(offspring)[idx_best]) # select the mu best ballots
        argmax = idx_best[-1]
        best_ranking = offspring[argmax]
        h.append(scores[argmax]) # collect best score
    r = process_vote(m, np.asarray(best_ranking), axis=1-axis)
    if verbose:
        rk.show_learning_curve(h)
        print('Best centrality score: {}'.format(h[-1]))
    if history:
        return r, h
    return r

def center(m, axis=1, method='euclidean', verbose=True, **kwargs):
    """ Optimal rank aggregation: find the ranking with the best centrality.

    This is the geometric median (or 1-center) of the judges for the given
    metric, found by differential evolution [Storn and Price, 1997]. The search
    is stochastic: pass `seed` to get reproducible results.

    Args:
        m: 2D matrix of scores.
        axis: Axis of judges.
        method: Distance or correlation used as metric (see `rk.centrality`).
        verbose: If True, print the optimizer termination message.
        **kwargs: Arguments passed to `scipy.optimize.differential_evolution` (e.g. `seed`, `maxiter`).
    """
    m_np = np.array(m)
    bounds = [(m_np.min(), m_np.max()) for _ in range(m_np.shape[1-axis])]
    res = differential_evolution(rk.mean_distance, bounds, (m_np, 1-axis, method), disp=False, **kwargs)
    if verbose:
        print(res.message)
    r = res.x
    return process_vote(m, r, axis=axis)


##########################
##### CONSENSUS ##########
##########################

def consensus(m, axis=0):
    """ Strict consensus between ranked ballots.

    Args:
        m: 2D matrix of ranks.
        axis: Axis of candidates.

    Returns:
        For each candidate, True if all judges agree on its value.
    """
    m_arr = np.array(m)
    if axis == 0:
        m_arr = m_arr.T
    r = np.all(m_arr == np.take(m_arr, 0, axis=0), axis=0)
    return process_vote(m, r, axis=1-axis)
