#######################################
### EVALUATION, COMPARISON, METRICS ###
#######################################

import numpy as np
from Levenshtein import distance as levenshtein
from scipy.spatial.distance import hamming
from scipy.stats import kendalltau, spearmanr, pearsonr
import ranky.ranking as rk
import itertools as it
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, jaccard_score, roc_auc_score, mean_squared_error

# METRICS #
# error bars
# bootstrap

CORR_METHODS = ['swap', 'kendalltau', 'spearman', 'spearmanr', 'pearson', 'pearsonr']
DIST_METHODS = ['hamming', 'levenshtein', 'winner', 'euclidean']

def to_dense(y):
    """ Format predictions/solutions from sparse format (1D) to dense format (2D).
        Sparse = 'argmax',
        Dense = 'one-hot'
    """
    y = np.array(y)
    if len(y.shape) == 1:
        length = y.shape[0]
        dense = np.zeros((length, y.max()+1))
        dense[np.arange(length), y] = 1
        return dense
    else:
        raise Exception('y must be 1-dimensional')

def to_sparse(y, axis=1):
    """ Format predictions/solutions from dense format (2D) to sparse format (1D).
        Sparse = 'argmax',
        Dense = 'one-hot'
    """
    y = np.array(y)
    if len(y.shape) == 2:
        sparse = np.argmax(y, axis=axis)
        return sparse
    else:
        raise Exception('y must be 2-dimensional')

def to_binary(y, threshold=0.5):
    """ Format predictions/solutions from probabilities to binary.
        1 if the value is stricly greater than the threshold, 0 otherwise.
    """
    return np.where(y > 0.5, 1, 0)

def metric(y_true, y_pred, method='accuracy', reverse_loss=False, missing_score=-1):
    """ Compute a classification scoring metric between y_true and y_pred.

        :param y_true: Ground truth (format?)
        :param y_pred: Predictions (format?)
        :param method: Name of the metric. Metrics available: 'accuracy', 'balanced_accuracy', 'precision', 'average_precision', 'brier', 'f1_score', 'mxe', 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse'
        :param reverse_loss: If True, return (1 - score).
        :param missing_score: Value to return if the computation fails.

        Format predictions AutoDL
        [[0.2, 0.3, 0.5]
        [0.1, 0.8, 0.1]]
        ...

        Ground truth
        [[0, 0, 1]
        [0, 1, 0]]
    """
    # TODO: Lift, BEP (precision/recall break-even point), Probability Calibration, Average recall, (SAR)
    # MISSING SCORE
    score = -1
    # PARAMETERS
    average = 'binary'
    if y_true.shape[1] > 2: # target is not binary
        average = 'micro'
    # PREPROCESSING
    if method in ['accuracy', 'balanced_accuracy', 'precision', 'f1_score', 'recall', 'jaccard']: # convert to sparse and binarize with 0.5 threshold
        y_true, y_pred = to_binary(y_true), to_binary(y_pred)
        y_true, y_pred = to_sparse(y_true), to_sparse(y_pred)
    elif method in ['mse', 'rmse']: # sparse format but keep probabilities
        y_true, y_pred = to_sparse(y_true), to_sparse(y_pred)
    # COMPUTE SCORE
    if method == 'accuracy':
        score = accuracy_score(y_true, y_pred)
    elif method == 'balanced_accuracy':
        score = balanced_accuracy_score(y_true, y_pred)
    elif method == 'precision':
        score = precision_score(y_true, y_pred, average=average)
    elif method == 'average_precision':
        score = average_precision_score(y_true, y_pred)
    elif method == 'f1_score':
        score = f1_score(y_true, y_pred, average=average)
    elif method == 'mxe':
        score = log_loss(y_true, y_pred)
    elif method == 'recall':
        score = recall_score(y_true, y_pred, average=average)
    elif method == 'jaccard':
        score = jaccard_score(y_true, y_pred, average=average)
    elif method == 'roc_auc':
        score = roc_auc_score(y_true, y_pred)
    elif method == 'mse':
        score = mean_squared_error(y_true, y_pred)
    elif method == 'rmse':
        score = mean_squared_error(y_true, y_pred, squared=False)
    elif method == 'sar':
        score = combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rmse'], method='mean')
    else:
        raise Exception('Unknown method: {}'.format(method))
    # REVERSE LOSS
    is_loss = method in ['mxe', 'mse', 'rmse']
    if reverse_loss and (score != missing_score) and is_loss:
        score = 1 - score
    return score

def combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rmse'], method='mean'):
    """ Combine several metrics as one.
        For example, you can compute SAR metric by calling:
        combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rms'])

        :param metrics: List of metric names.
        :param ranking: A ranking system from ranky.
    """
    score = -1
    scores = [metric(y_true, y_pred, m, reverse_loss=True) for m in metrics]
    if method == 'mean':
        score = np.mean(scores)
    elif method == 'median':
        score = np.median(scores)
    else:
        raise Exception('Unknwon method: {}'.format(method))
    return score

def dist(r1, r2, method='hamming'):
    """ Levenshtein/Wasserstein type distance between two ranked ballots.
        0, 1

        :param method: 'hamming', 'levenshtein', 'winner', 'euclidean', 'winner_mistake'.
    """
    # https://math.stackexchange.com/questions/2492954/distance-between-two-permutations
    # https://people.revoledu.com/kardi/tutorial/Similarity/OrdinalVariables.html
    # L1 norm between permutation matrices (does it work with ties?)
    # Normalized Rank Transformation
    # Footrule distance
    # Damareau-Levenshtein - transposition distance
    # Cayley distance - Kendall but with any pairs
    # Ulam / LCS distance - number of delete-shift-insert operations (no ties)
    # Chebyshev /maximum distance
    # Minkowski distance
    # Jaro-Winkler distance - only transpositions
    if method == 'hamming': # Hamming distance: number of differences
        d = hamming(r1, r2)
    elif method == 'levenshtein': # Levenshtein distance - deletion, insertion, substitution
        d = levenshtein(arr_to_str(r1), arr_to_str(r2))
    elif method == 'winner': # How much the ranked first in r1 is far from the first place in r2
        i = np.argmin(r1)
        d = r2[i] - r1[i] # TODO: should be an absolute value?
    elif method == 'euclidean':
        if not isinstance(r1, np.ndarray):
            r1, r2 = np.array(r1), np.array(r2)
        d = np.linalg.norm(r1 - r2)
    elif method == 'winner_mistake': # 0 if the winner is the same (TODO: ties?)
        d = 1
        if np.argmin(r1) == np.argmin(r2):
            d = 0
    else:
        raise(Exception('Unknown distance method: {}'.format(method)))
    return d

def corr(r1, r2, method='swap', return_p_value=False):
    """ Levenshtein/Wasserstein type correlation between two ordinal distributions.
        -1, 0, 1

        :param method: 'swap', 'spearman', 'pearson'
        :param p_value: If True, return a tuple (score, p_value)
    """
    if method in ['swap', 'kendalltau']: # Kendalltau: swap distance
        c, p_value = kendalltau(r1, r2)
    elif method in ['spearman', 'spearmanr']: # Spearman rank-order
        c, p_value = spearmanr(r1, r2)
    elif method in ['pearson', 'pearsonr']: # Pearson correlation
        c, p_value = pearsonr(r1, r2)
    else:
        raise(Exception('Unknown correlation method: {}'.format(method)))
    if return_p_value:
        return c, p_value
    return c

def kendall_w(matrix, axis=0):
    """ Kendall's W coefficient of concordance.
        /!/ No correction for ties.

        :param axis: Axis of judges.
    """
    matrix = rk.rank(matrix, axis=1-axis) # compute on ranks
    m = matrix.shape[axis] # judges
    n = matrix.shape[1-axis] # candidates
    denominator = m**2 * (n**3 - n)
    rating_sums = np.sum(matrix, axis=axis)
    S = n * np.var(rating_sums)
    return 12 * S / denominator

def kendall_w_ties(matrix, axis=0):
    """ Kendall's W coefficient of concordance.
        /!\ STILL TODO

        :param axis: Axis of raters.
    """
    # TODO kendall W with tie correction
    # https://en.wikipedia.org/wiki/Kendall%27s_W
    matrix = rk.rank(matrix, axis=1-axis) # compute on ranks
    m = matrix.shape[axis] #raters
    n = matrix.shape[1-axis] # items rated
    denominator = m**2 * (n**3 - n)
    rating_sums = np.sum(matrix, axis=axis)
    S = n * np.var(rating_sums)
    return 12 * S / denominator

def concordance(m, method='spearman', axis=0):
    """ Coefficient of concordance between ballots.
        This is a measure of agreement between raters.
        The computation is the mean of the correlation between all possible pairs of judges.

        :param axis: Axis of raters.
    """
    # Idea: Kendall's W - linearly related to spearman between all pairwise
    if rk.is_dataframe(m):
        m = np.array(m)
    idx = range(m.shape[axis])
    scores = []
    for pair in it.permutations(idx, 2):
        r1 = np.take(m, pair[0], axis=axis)
        r2 = np.take(m, pair[1], axis=axis)
        c, p_value = corr(r1, r2, method=method, return_p_value=True)
        scores.append(c)
    return np.mean(scores)

def auc_step(X, Y):
    """ Compute area under curve using step function (in 'post' mode).

        :param X: List of timestamps of size n
        :param Y: List of scores of size n
    """
    # Log scale
    def transform_time(t, T=1200, t0=60):
        return np.log(1 + t / t0) / np.log(1 + T / t0)
    X = [transform_time(t) for t in X]
    # Add origin and final point
    X.insert(0, 0)
    Y.insert(0, 0)
    X.append(1)
    Y.append(Y[-1])
    if len(X) != len(Y):
        raise ValueError("The length of X and Y should be equal but got " +
                         "{} and {} !".format(len(X), len(Y)))
    # Compute area
    area = 0
    for i in range(len(X) - 1):
        delta_X = X[i + 1] - X[i]
        area += delta_X * Y[i]
    return area

#TODO
#def relative_consensus or consensus_graph

def quality(m, r, axis=0, method='swap'):
    """ Compute how good a ranking is by doing the sum of the correlations between the ranking and all ballots in m.
        Also called centrality.
        :param method: 'hamming', 'levenshtein' for distance. 'swap', 'spearman' for correlation.

        TODO: pairwise corr/dist between all ballots?
    """
    if method in CORR_METHODS: # correlation
        scores = np.apply_along_axis(corr, axis, m, r, method) # best 1
    else: # distance
        scores = - np.apply_along_axis(dist, axis, m, r, method) # minus because higher is better, best 0
    return scores.mean()

def mean_distance(r, m, axis, method):
    """ Mean distance between r and all points in m.
    """
    return - quality(m, r, axis=axis, method=method)

def correct_metric(metric, model, X_test, y_test, average='weighted', multi_class='ovo'):
    """ Compute the model's score by making predictions on X_test and comparing them with y_test.
        Try different configuration to be robust to all sklearn metrics.
    """
    ### /!\ TODO: CLEAN CODE BELOW /!\ ###
    # TODO: Vector case and one-hot case
    try:
        y_pred = model.predict_proba(X_test) # SOFT
        try:
            score = metric(y_test, y_pred, average=average, multi_class='ovo') #labels=np.unique(y_pred))
        except:
            try:
                score = metric(y_test, y_pred, average=average)
            except:
                score = metric(y_test, y_pred)
    except:
        y_pred = model.predict(X_test) # HARD
        try:
            score = metric(y_test, y_pred, average=average, multi_class='ovo')
        except:
            try:
                score = metric(y_test, y_pred, average=average)
            except:
                try:
                    score = metric(y_test, y_pred)
                except:
                    labels = np.unique(y_pred)
                    score = metric(y_test, y_pred, labels=labels)
    return score
