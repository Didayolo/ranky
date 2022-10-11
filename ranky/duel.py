# Module for pairwise comparison of performances of algorithm

################################################
####     Metrics for pairwise methods       ####
####       and significance tests           ####
################################################

# TODO: clarify names, add scored version of NHST and more.

import numpy as np
from scipy.stats import binom_test
from baycomp import two_on_single, two_on_multiple

def hard_wins(a, b, reverse=False):
    """ Function returning True if a wins against b in a majority vote.

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
    """ Function returning 1 if a wins against b in a majority vote, 0.5 in case of a tie and 0 otherwise.

    Useful for to compute Copeland's method.

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
    """ Function returning True if a significantly wins against b.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        pval: A win is counted only if the probability of the null hypothesis (tie) is equal or smaller than pval.
                     If pval is set to 1, then p_wins is equivalent to hard_wins function.
        reverse: If True, lower is better.
    """
    a, b = np.array(a), np.array(b)
    Wa, Wb = np.sum(a > b), np.sum(b > a)
    if reverse:
        Wa, Wb = np.sum(a < b), np.sum(b < a)
    significant = binom_test(Wa, n=len(a), p=0.5) <= pval
    wins = Wa > Wb
    return significant and wins # count only significant wins

def bayes_wins(a, b, width=0.1, independant=False, score=False):
    """ Compare a and b using a Bayesian signed-ranks test.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        width: the width of the region of practical equivalence.
        independant: True if the different scores are correlated (e.g. bootstraps or cross-validation scores).
        score: If True, returns the probability of winning instead of a boolean.
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
    """ Alias for bayes_wins but returning probability of winning.
    """
    return bayes_wins(a, b, score=True, **kwargs)

def success_rate(a, b, reverse=False, ties=False):
    """ Returns the frequency (rate) of a > b.

    Args:
        a: Ballot representing one candidate (array-like).
        b: Ballot representing one candidate (array-like).
        reverse: If True, lower is better.
        ties: If True, ties are taken into account (with value 0.5) instead of hard comparisons
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
    """ Returns the mean relative difference between a and b.

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
################################################
################################################
