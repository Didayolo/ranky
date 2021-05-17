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
#import ranky.metric as metric
import ranky as rk

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

# upsampling
# downsampling

def is_series(m):
    return isinstance(m, pd.Series)

def is_dataframe(m):
    return isinstance(m, pd.DataFrame)

def process_vote(m, r, axis=1):
    """ Keep names if using pd.DataFrame.
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
    """
    voter = np.random.randint(m.shape[axis]) # select a column number
    r = np.take(np.array(m), voter, axis=axis) #m[:, voter]
    return process_vote(m, r, axis=axis)

def borda(m, axis=1, method='mean', reverse=False):
    """ Borda count.

    Args:
        method: 'mean' or 'median'.
        reverse: reverse the ranking.
    """
    ranking = rank(m)
    if reverse:
        ranking = rank(-m)
    if method == 'mean':
        r = ranking.mean(axis=axis)
    elif method == 'median':
        r = ranking.median(axis=axis)
    else:
        raise(Exception('Unknown method for borda system: {}'.format(method)))
    return process_vote(m, r, axis=axis)

def majority(m, axis=1):
    """ Majority judgement.
    """
    r = np.median(m, axis=axis)
    return process_vote(m, r, axis=axis)

def score(m, axis=1):
    """ Score/range ranking.
    """
    r = np.mean(m, axis=axis)
    return process_vote(m, r, axis=axis)

def uninominal(m, axis=1, turns=1):
    """ Uninominal.

    Args:
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
    This function is currently an alias for `rk.condorcet`.
    This is more sound to name this function "pairwise".
    Let's keep condorcet for a while for retro-compatibility but we need to clean this.

    Args:
        m: Preference matrix.
        axis: Judge axis.
        wins: Function returning True if a wins against b. `rk.hard_wins` used by default.
        return_graph: If True, returns the 1-1 matches result matrix.
        score: If True, produce the 'Condorcet score' by dividing the results by (n - 1)
               with n the number of candidates.
    """
    return condorcet(m, axis=axis, wins=wins, return_graph=return_graph, score=score, **kwargs)

def condorcet(m, axis=1, wins=None, return_graph=False, score=False, **kwargs):
    """ Condorcet method.

    Args:
        m: Preference matrix.
        axis: Judge axis.
        wins: Function returning True if a wins against b. `rk.hard_wins` used by default.
        return_graph: If True, returns the 1-1 matches result matrix.
        score: If True, produce the 'Condorcet score' by dividing the results by (n - 1)
               with n the number of candidates.
    """
    if wins is None:
        wins = rk.hard_wins
    ranking = rank(m, axis=1-axis)
    n = len(ranking)
    r = np.array(ranking)
    graph = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            graph[i][j] = wins(r[i], r[j], **kwargs)
    r = np.sum(graph, axis=1-axis)
    r = process_vote(m, r, axis=axis)
    if score:
        r = r / (n - 1)
    if return_graph:
        return r, graph
    return r

def copeland(m, axis=1, **kwargs):
    """ Copeland's method.

    This function is an alias of calling `rk.condorcet` function with `rk.copeland_wins` as the wins function.

    Args:
        m: Preference matrix.
        axis: Judge axis.
        **kwargs: Arguments to be passed to `rk.condorcet` function.
    """
    return condorcet(m, axis=axis, wins=rk.copeland_wins, **kwargs)

def kemeny_young(m, axis=1, **kwargs):
    """ Kemeny-Young method.

    This function is an alias of calling `rk.center` function with Kendall tau as the metric.
    Indeed, Kemeny-Young method consists in computing the ranking that is the closest to all judges,
    according to Kendall's distance.

    Args:
        m: Preference matrix.
        axis: Judge axis.
        **kwargs: Arguments to be passed to `rk.center` function.
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
    res = differential_evolution(rk.mean_distance, bounds, (m_np, 1-axis, method), disp=False) # from scipy.optimize
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

##########################
###### RCV RANKING #######
##########################

'''
Takes in only a pandas dataframe as input
Format of input dataframe: ranked values from 1 to n, where n is number of classes
returns: winning class by rcv (None if there was a tie), ordered list of eliminated classes

This requires a dataframe in this format. It would be theoretically possible to implement other formats but this just makes no sense for rcv voting.
This is because rcv voting relies on a list of preferences from first to last, so using other input formats is just senseless

This has been tested on the data the function was written for, but not exhaustively tested
The code should work for ties but hasn't been tested on an dataset where rcv does result in ties at an intermediate step

Note on this contribution: (feel free to delete if you approve pull request)
    The logic i followed to implement RCV is below, in pseudocode. it should help you understand what the code is doing easier

        Terms:
            Voters: rankers (row in dataframe)
            Classes: different choices every voter has (column in dataframe)
            Majority: when one choice is the preference of more than 50% of Voters

        Count first preference of every voter
        If there is a majority, return that class
        Else Repeat until one class is a majority:
            identify the minority class
            eliminate the minority class from the vote
            for every voter:
                if their first preference is a class in the running, do nothing
                if their first preference is a class that just got eliminated, find their next valid preference and count their vote for them
                if thier first preference is a class that was eliminated in a previous round, check their next valid preference
                    if thier next valid preference just got eliminated, find a further valid preference and count their vote for them
                    if their next valid prefernece is in the running, do nothing [their vote got counted for that preference before]
            
'''

def rcv(df):
    classes = df.shape[0]       # dataframe classes
    votes = {}                  # votes for each class, as calculated below
    indices = df.index
    minority_classes = []       # all eliminated classes
    demoted_classes = []        # most recent set of eliminated classes
    elim = []                   # classes in order of elimination

    # initialize votes dictionary
    for _class in range(classes):
        votes[indices[_class]] = 0

    # compute total number of voters, and majority
    total_votes = len(df.columns)
    majority = (total_votes / 2) + 1
    
    # loop up till number of classes
    # this is unecessary but should avoid infinite loops in case something in the dataset is broken
    for iteration in range(classes):

        # iterate over voters
        for c in df:
            col = df[c]

            # get class of first preference for that voter
            ind_class = col.index[col == 1][0]

            # check if class was demoted previously
            if ind_class in demoted_classes:
                # find next preference class that has not been eliminated and assign vote to it
                for i in range(2, classes + 1):
                    next_ind_class = col.index[col == i][0]
                    if next_ind_class not in minority_classes:
                        votes[next_ind_class] += 1
                        break

            # check if it was not just demoted but was at some previous time
            # ex: this is for the case a voter who was moved to their second preference has to be moved to their third preference now
            elif ind_class in minority_classes:
                # iterate over total votes
                i = 2
                while i < classes + 1:
                    # find the next preference of the voter
                    next_ind_class = col.index[col == i][0]

                    # if the preference was just eliminated last round, then assign vote to the next valid preference
                    # if it was eliminated in the past, continue searching
                    # if it was never eliminated, then exit the loop because otherwise we would double-count votes
                    if next_ind_class in demoted_classes:
                        i += 1
                        while i < classes + 1:
                            next_ind_class = col.index[col == i][0]
                            if next_ind_class not in minority_classes:
                                votes[next_ind_class] += 1
                                break
                            i += 1
                        break
                    elif next_ind_class in minority_classes:
                        i += 1
                    else:
                        break

            # if this is the first iteration of the outermost loop, we have to count votes for the first time based on first preference
            elif iteration == 0:
                votes[ind_class] += 1

        # iterate over classes
        for index in indices:
            # skip a class that was previously eliminated
            if index in minority_classes:
                continue

            # check if there is a new majority class
            if votes[index] > majority:
                vals = list(votes.values())
                vals.sort()

                # eliminate the remaining classes in order of least to most share of votes
                for val in vals[:-1]:
                    key = ([key for key in votes if votes[key] == val])[0]
                    elim.append(key)
                    del votes[key]

                # return the winning class and all eliminated classes in order
                return index, elim

        # there was no majority, so find the minority class(es)
        min_val = min(votes.values())
        keys = [key for key in votes if votes[key] == min_val]
        # mark minority classes as recently eliminated in the last round
        demoted_classes = keys
        # mark minority classes as eliminated
        minority_classes = minority_classes + keys
        elim += keys
        # delete minority classes from the vote
        for key in keys:
            del votes[key]
        # if there are no classes left in the vote, that means there was a tie
        # there is no winner, so return None
        if len(votes) == 0:
            return(None, elim)