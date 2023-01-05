#######################
### RANKING SYSTEMS ###
#######################

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
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
    """ Replace values by their rank in the column.

    By default, higher is better.
    TODO: save parameters to add values to an already fitted ranking.

    Args:
        m: Preference matrix.
        axis: Candidates axis.
        method: 'average', 'min', 'max', 'dense', 'ordinal'
        ascending: Ascending or descending order.
        reverse: Reverse order.

    Returns:
        np.ndarray, pd.Series, pd.DataFrame
        Ranked preference matrix.
    """
    if isinstance(m, list):
        m = np.array(m)
    if ascending == reverse: # greater is better (descending order)
        m = -m # take the opposite to inverse rank
    r = np.apply_along_axis(rankdata, axis, m, method=method) # convert values to ranking in all rows or columns
    return process_vote(m, r)

def weigher(r, method='hyperbolic'):
    """ The weigher function.

    Must map nonnegative integers (zero representing the most important element) to a nonnegative weight.
    The default method, 'hyperbolic', provides hyperbolic weighing, that is, rank r is mapped to weight 1/(r+1)

    Args:
        r: Integer value (supposedly a rank) to weight
        method: Weighing method. 'hyperbolic'
    """
    if method == 'hyperbolic':
        return 1 / (r + 1)
    else:
        raise Exception('Unknown method: {}'.format(method))

def tie(r, threshold=0.1):
    """ TODO: merge close values.
    """
    return 0

def contains_ties(r):
    """ Return True if r contains tied values.

    Args:
        r: 1D array-like of scores or ranks.
    """
    n = len(r)
    if n==0:
        return False
    for i in range(n):
        for j in range(1, n):
            if i != j:
                if r[i] == r[j]:
                    return True
    return False

# remove rows (candidates) or columns (voters)
def bootstrap(m, axis=0, n=None, replace=True, return_holdout=False):
    """ Sample with replacement among an axis (and keep the same shape by default).

    By convention rows reprensent candidates and columns represent voters.

    Args:
        axis: Axis concerned by bootstrap.
        n: Number of examples to sample. By default it is the size of the matrix among the axis.
        replace: Sample with or without replacement. It is not bootstrap if the sampling is done without replacement.
        return_holdout: If True, returns a tuple (bootstrap, out-of-bag set).
    """
    if n is None:
        n = m.shape[axis]
    idx = np.random.choice(m.shape[axis], n, replace=replace)
    bootstrap = np.take(m, idx, axis=axis)
    if return_holdout:
        ran = np.arange(m.shape[axis])
        holdout_idx = ran[np.array([x not in idx for x in ran])]
        holdout = np.take(m, holdout_idx, axis=axis)
        return bootstrap, holdout
    else:
        return bootstrap

def joint_bootstrap(m_list, axis=0, n=None, replace=True):
    """ Apply the same bootstrap on all matrices on m_list.

    Sample with replacement among an axis (and keep the same shape by default).
    By convention rows reprensent candidates and columns represent voters.

    Args:
        axis: Axis concerned by bootstrap.
        n: Number of examples to sample. By default it is the size of the matrix among the axis.
        replace: Sample with or without replacement. It is not bootstrap if the sampling is done without replacement.
    """
    size = m_list[0].shape[axis]
    if n is None:
        n = size
    idx = np.random.choice(size, n, replace=replace)
    return [np.take(m, idx, axis=axis) for m in m_list]

def top_k_method(D, F, k=1, reverse=False):
    """ Apply top-k method to select a winner from two rankings D and F (development and final).

    Return the index of the winner. The winner is the top of F from the k best candidates of D.

    Args:
        D: 1D array-like of scores, representing the development phase (or public leaderboard).
        F: 1D array-like of scores, representing the final phase (or private leaderboard).
        k: number of candidates that access the final phase.
        reverse: if True lower is better (by default higher is better).
    """
    D, F = to_series(D), to_series(F)
    top_k = select_k_best(D, k=k, reverse=reverse)
    return rk.select_best(F[top_k], reverse=reverse)

def select_k_best(m, k=1, reverse=False):
    """ Select k best candidates from the 1D array m.

    Args:
        m: 1D array-like of scores.
        k: number of best candidates to be returned.
        reverse: if True lower is better (by default higher is better).
    """
    m = to_series(m)
    if k == 0 or k > len(m):
        raise Exception('Bad value for K')
    return m.sort_values(ascending=reverse).index[:k]

def select_best(m, reverse=False):
    """ Return the best candidate from the 1D array m.

    Args:
        m: 1D array-like of scores.
        reverse: if True lower is better (by default higher is better).
    """
    return select_k_best(m, k=1, reverse=reverse)[0]

# upsampling
# downsampling

def is_series(m):
    return isinstance(m, pd.Series)

def is_dataframe(m):
    return isinstance(m, pd.DataFrame)

def to_series(m):
    if not isinstance(m, list):
        if len(m.shape) == 2 and m.shape[1] == 1: # "column array"
            m = m.reshape((m.shape[0]))
    if not is_series(m): # cast to pd.Series if needed
        m = pd.Series(m)
    return m

def process_vote(m, r, axis=1):
    """ Keep names if using pd.DataFrame.

        Args:
            m: original matrix of scores (pd.DataFrame or pd.Series)
            r: the ranking (array-like)
    """
    if is_dataframe(m):
        if len(r.shape) == 1: # Series
            if axis==0: # Voting axis
                r = pd.Series(r, m.columns) # Participants names
            elif axis==1:
                r = pd.Series(r, m.index)
        elif len(r.shape) == 2: # DataFrame
            r = pd.DataFrame(r, index=m.index, columns=m.columns)
    elif is_series(m): # From Series to Series
        r = pd.Series(r, m.index)
    return r

#################################
####### RANKING SYSTEMS #########
#################################

#################################
##### 1. CLASSICAL METHODS #######
#################################

def dictator(m, axis=1):
#def random(m, axis=1): # renamed because of random module
    """ Random dictator.

        Args:
            m: 2D matrix of scores.
            axis: axis of judges.
    """
    voter = np.random.randint(m.shape[axis]) # select a column number
    r = np.take(np.array(m), voter, axis=axis) #m[:, voter]
    return process_vote(m, r, axis=axis)

def borda(m, axis=1, method='mean', reverse=False):
    """ Borda count.

    Args:
        m: 2D matrix of scores.
        axis: axis of judges.
        method: 'mean' or 'median'.
        reverse: reverse the ranking.
    """
    ranking = rank(m, axis=1-axis)
    if reverse:
        ranking = rank(-m, axis=1-axis)
    if method == 'mean':
        r = ranking.mean(axis=axis)
    elif method == 'median':
        r = ranking.median(axis=axis)
    else:
        raise(Exception('Unknown method for borda system: {}'.format(method)))
    return process_vote(m, r, axis=axis)

def majority(m, axis=1):
    """ Majority judgement.

        Args:
            m: 2D matrix of scores.
            axis: axis of judges.
    """
    r = np.median(m, axis=axis)
    return process_vote(m, r, axis=axis)

def score(m, axis=1):
    """ Score/range ranking.

        Args:
            m: 2D matrix of scores.
            axis: axis of judges.
    """
    r = np.mean(m, axis=axis)
    return process_vote(m, r, axis=axis)

def uninominal(m, axis=1, turns=1):
    """ Uninominal.

    Args:
        m: 2D matrix of scores.
        axis: axis of judges.
        turns: number of turns.
    """
    if turns > 1:
        raise(Exception('Uninominal system is currently implemented only in one turn setting.'))
    ranking = rank(m, axis=1-axis) # convert to rank
    #for turn in range(turns): # TODO replace 1 by turn in next line
    r = (ranking == 1).sum(axis=axis)  # count number of uninominal vote (first in judgement)
    return process_vote(m, r, axis=axis)

def pairwise(m, axis=1, wins=None, return_graph=False, score=False, **kwargs):
    """ Pairwise method.

    We compute the matrix of scores of all possible pairs of matches between all candidates.
    The score of one match is defined by the `wins` function.

    Args:
        m: 2D matrix of scores (preference matrix).
        axis: Judge axis. /!\
        wins: Function returning True if a wins against b. `rk.copeland_wins` used by default.
        return_graph: If True, returns the 1-1 matches result matrix.
        score: If True, produce scores between 0 and 1 by dividing the results by (n - 1)
               with n the number of candidates.
    """
    if wins is None:
        wins = rk.copeland_wins
    _m = m
    m = np.array(m)
    n = m.shape[1-axis]
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                c1, c2 = np.take(m, i, 1-axis), np.take(m, j, 1-axis)
                graph[i][j] = wins(c1, c2, **kwargs)
            else:
                graph[i][j] = 0 # no comparison with itself
    r = np.sum(graph, axis=1) # collect candidates average score against all opponents
    r = process_vote(_m, r, axis=axis)
    if score:
        r = r / (n - 1)
    if return_graph:
        return r, graph
    return r

def copeland(m, axis=1, **kwargs):
    """ Copeland's method.

    This function is an alias of calling `rk.pairwise` function with `rk.copeland_wins` as the wins function.

    Args:
        m: 2D matrix of scores.
        axis: axis of judges.
        **kwargs: Arguments to be passed to `rk.pairwise` function.
    """
    return pairwise(m, axis=axis, wins=rk.copeland_wins, **kwargs)

def kemeny_young(m, axis=1, **kwargs):
    """ Kemeny-Young method.

    This function is an alias of calling `rk.center` function with Kendall tau as the metric.
    Indeed, Kemeny-Young method consists in computing the ranking that is the closest to all judges,
    according to Kendall's distance.

    Args:
        m: 2D matrix of scores.
        axis: axis of judges.
        **kwargs: arguments to be passed to `rk.center` function.
    """
    return center(m, axis=axis, method='kendalltau', **kwargs)


#################################
### 2. COMPUTATIONAL METHODS ####
#################################

# Based on optimization.

def brute_force(m, axis=0, method='swap'):
    """ Brute force search.

    TODO:
        - keep all optimal solutions.
        - ties
        - docs
    """
    best_score = -1
    best_r = np.take(m, 0, axis=axis)
    for r in tqdm(list(it.permutations(range(m.shape[axis])))): # all possible rankings
        score = centrality(m, r, axis=axis, method=method)
        if score > best_score:
            best_score = score
            best_r = r
    return best_r

def random_swap(r, n=1, tie=0.1):
    """ Swap randomly two values in r.

    Used by evolution_strategy function.

    Args:
        n: number of consecutive changes
        tie: probability of tie instead of swap
    """
    _r = r.copy()
    for _ in range(n):
        i1, i2 = np.random.randint(len(r)), np.random.randint(len(r))
        if tie!=0 and np.random.randint(1/tie) == 0: # generate tie with some probability
            _r[i1] = _r[i2]
        else: # swap two values
            _r[i1], _r[i2] = _r[i2], _r[i1]
    return _r

def evolution_strategy(m, axis=0, mu=10, l=2, epochs=50, n=1, tie=0.1, method='swap', history=False, verbose=False):
    """ Use evolution strategy to search the best centrality ranking.

    Return the best ranking (and the best score of each generation if needed).

    Args:
        axis: candidates axis.
        mu: population size.
        l: mu * l = offspring size.
        epochs: number of iterations.
        n: number of swaps performed during a single mutation.
        tie: probability of performing a tie instead of a swap during mutation process.
        method: method used to compute centrality of the ranking.
        history: if True, return a tuple (ranking, history).
        verbose: if True, plot the learning curve.
    """
    r = np.arange(m.shape[axis])
    h = []
    population = [sorted(r, key=lambda k: _random()) for _ in range(mu)] # mu random ranked ballots
    best_ranking = population[0] # initialize best_ranking
    for epoch in tqdm(range(epochs)):
        offspring = [random_swap(x, n=n, tie=tie) for x in population*l] # random swaps to generate new ranked ballots
        offspring.append(best_ranking) # add the previous best to the offspring to avoid losing it if no children beat it
        scores = [centrality(m, child, axis=axis, method=method) for child in offspring] # compute fit function
        idx_best = np.argsort(scores)[len(scores)-mu:]
        population = list(np.array(offspring)[idx_best]) # select the mu best ballots
        argmax = idx_best[-1]
        best_ranking = offspring[argmax]
        h.append(scores[argmax]) # collect best score
    r = process_vote(m, best_ranking, axis=1-axis)
    if verbose:
        show_learning_curve(h)
        print('Best centrality score: {}'.format(h[-1]))
    if history:
        return r, h # return the best ranking and its score
    return r

def center(m, axis=1, method='euclidean', verbose=True, **kwargs):
    """ Find the geometric median or 1-center.

    Solve the metric facility location problem.
    Find the ranking maximizing the centrality.
    Also called optimal rank aggregation.

    Args:
        axis: judges axis.
        method: distance or correlation method used as metric.
        verbose: show optimization termination message.
        **kwargs: arguments for `scipy.differential_evolution` function.
    """
    # Finds the global minimum of a multivariate function.
    # Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimium, and can search large areas of candidate space,
    # but often requires larger numbers of function evaluations than conventional gradient based techniques.
    # The algorithm is due to Storn and Price [R150].
    m_np = np.array(m)
    bounds = [(m_np.min(), m_np.max()) for _ in range(m_np.shape[1-axis])]
    res = differential_evolution(rk.mean_distance, bounds, (m_np, 1-axis, method), disp=False, **kwargs) # from scipy.optimize
    if verbose:
        print(res.message)
    r = res.x
    return process_vote(m, r, axis=axis)


##########################
##### CONSENSUS ##########
##########################

def consensus(m, axis=0):
    """ Strict consensus between ranked ballots.
    """
    m_arr = np.array(m)
    if axis==0:
        m_arr = m_arr.T
    r = np.all(m_arr == np.take(m_arr, 0, axis=0), axis=0)
    return process_vote(m, r, axis=1-axis)


# STATISTICAL TESTS #
# McNemar test
# statistic = (Yes/No - No/Yes)^2 / (Yes/No + No/Yes)

# Friedman test
