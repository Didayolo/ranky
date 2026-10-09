################################################
####     Metrics for pairwise methods       ####
####       and significance tests           ####
################################################

# Functions f(a, b) comparing two candidates, where a and b are the scores of
# each candidate given by the same judges. They can be passed as the `wins`
# argument of `rk.pairwise` or as `comparison_func` of `rk.critical_difference`.

# TODO: clarify names, add scored version of NHST and more.

import numpy as np
from scipy.stats import binomtest
from baycomp import two_on_single, two_on_multiple

def declare_ties(a, b, comparison_func=None, **kwargs):
    """ Return True if a and b are tied according to the comparison function.

    They are tied if comparison_func(a, b) == comparison_func(b, a), i.e. if
    neither beats the other (or, in theory, both do).

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        comparison_func: Asymmetrical function used to compare two candidates.
            comparison_func(a, b) should return 1 (or True) if a beats b and 0 otherwise.
            By default it is `p_wins`, performing a binomial test.
        **kwargs: Arguments passed to the comparison function.
    """
    if comparison_func is None:
        comparison_func = p_wins
    return comparison_func(a, b, **kwargs) == comparison_func(b, a, **kwargs)

def hard_wins(a, b, reverse=False):
    """ Return True if a wins against b in a majority vote.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        reverse: If True, lower is better.
    """
    a, b = np.array(a), np.array(b)
    Wa, Wb = np.sum(a > b), np.sum(b > a)
    if reverse:
        Wa, Wb = np.sum(a < b), np.sum(b < a)
    return Wa > Wb  # hard comparisons

def copeland_wins(a, b, reverse=False):
    """ Return 1 if a wins against b in a majority vote, 0.5 in case of a tie and 0 otherwise.

    Used to compute Copeland's method.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        reverse: If True, lower is better.
    """
    a, b = np.array(a), np.array(b)
    Wa, Wb = np.sum(a > b), np.sum(b > a)
    if reverse:
        Wa, Wb = np.sum(a < b), np.sum(b < a)
    if Wa > Wb: # hard comparisons
        return 1
    elif Wb > Wa:
        return 0
    else: # Copeland's method
        return 0.5

def p_wins(a, b, pval=0.05, reverse=False):
    """ Return True if a significantly wins against b (two-sided binomial test).

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        pval: A win is counted only if the p-value of the test is lower than or equal to pval.
            With pval=1, p_wins is equivalent to `hard_wins`.
        reverse: If True, lower is better.
    """
    a, b = np.array(a), np.array(b)
    Wa, Wb = np.sum(a > b), np.sum(b > a)
    if reverse:
        Wa, Wb = np.sum(a < b), np.sum(b < a)
    significant = binomtest(Wa, n=len(a), p=0.5).pvalue <= pval
    wins = Wa > Wb
    return significant and wins # count only significant wins

def bayes_wins(a, b, width=0.1, independant=False, score=False):
    """ Compare a and b with a Bayesian test (using the `baycomp` package).

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        width: Width of the region of practical equivalence (rope).
        independant: If True, the scores are considered independent (e.g. scores on
            different datasets) and a Bayesian signed-rank test is used. If False,
            they are considered correlated (e.g. cross-validation folds on the
            same dataset) and a Bayesian correlated t-test is used.
        score: If True, return the probability that a wins instead of a boolean.

    Returns:
        True if "a wins" is the most probable outcome (among a wins, tie and b wins).
    """
    a, b = np.array(a), np.array(b)
    if independant:
        p_a, p_tie, p_b = two_on_multiple(a, b, rope=width)
    else:
        p_a, p_tie, p_b = two_on_single(a, b, rope=width)
    if score:
        res = p_a
    else:
        res = p_a == max([p_a, p_tie, p_b])
    return res

def bayes_score(a, b, **kwargs):
    """ Probability that a wins against b. Alias of `bayes_wins(a, b, score=True)`.
    """
    return bayes_wins(a, b, score=True, **kwargs)

def success_rate(a, b, reverse=False, ties=False):
    """ Return the frequency of a > b.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        reverse: If True, lower is better.
        ties: If True, ties count as half a win.
    """
    a, b = np.array(a), np.array(b)
    if not reverse: # normal behavior
        Wa = np.sum(a > b)
    else:
        Wa = np.sum(a < b)
    if ties:
        Eq = np.sum(a == b)
        Wa = Wa + Eq * 0.5
    return Wa / len(a) # hard comparisons

def relative_difference(a, b, reverse=False):
    """ Return the mean relative difference between a and b, (a - b) / (a + b).

    Pairs where a + b == 0 count as 0.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        reverse: If True, lower is better.
    """
    a, b = np.array(a, dtype='float'), np.array(b, dtype='float')
    if reverse:
        num = b - a
    else:
        num = a - b
    denom = a + b
    s = np.divide(num, denom, out=np.zeros_like(num), where=denom!=0)
    return np.mean(s)
