#######################################
### EVALUATION, COMPARISON, METRICS ###
#######################################
# Metrics, error bars, bootstrap

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein
from scipy.spatial.distance import hamming
from scipy.stats import kendalltau, spearmanr, pearsonr
import ranky.ranking as rk
import itertools as it
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, jaccard_score, roc_auc_score, mean_squared_error, mean_absolute_error

METRIC_METHODS = ['accuracy', 'balanced_accuracy', 'precision', 'average_precision', 'brier', 'f1_score', 'mxe', 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse', 'sar', 'mae']
CORR_METHODS = ['swap', 'kendalltau', 'spearman', 'spearmanr', 'pearson', 'pearsonr']
DIST_METHODS = ['hamming', 'levenshtein', 'winner', 'euclidean', 'winner_distance', 'asymmetrical_winner_distance']

def arr_to_str(a):
    return "".join(str(x) for x in a)

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

def to_binary(y, threshold=0.5, unilabel=False, at_least_one_class=False):
    """ Format predictions/solutions from probabilities to binary {0, 1}.

    Behaviour:
    If unilabel is False: 1 if the value is stricly greater than the threshold, 0 otherwise.
    If unilabel is True: argmax 1, other values 0.

    Args:
        y: vector or matrix to binarize. If unilabel is True or at_least_one_class is True, y must be in 2D dense probability format.
        threshold: threshold for binarization (0 if below, 1 if strictly above).
        unilabel: If True, return only one 1 for each row.
        at_least_one_class: If True, for each row, if no probability is above the threshold, the argmax is set to 1.
    """
    # TODO: keep index and column names if y is a pd.DataFrame
    y = np.array(y)
    if unilabel == True or at_least_one_class == True:
        if len(y.shape) != 2:
            raise Exception('If unilabel is True or at_least_one_class is True, y must be in 2D dense probability format.')
    n = y.shape[0]
    if unilabel:
        y_binary = np.zeros(y.shape, dtype=int)
        y_binary[np.arange(n), np.argmax(y, axis=1)] = 1
    else: # multi-label
        y_binary = np.where(y > threshold, 1, 0)
        if at_least_one_class:
            y_binary[np.arange(n), np.argmax(y, axis=1)] = 1
    return y_binary

def any_metric(a, b, method, **kwargs):
    """ Compute distance or correlation between a and b using any scoring metric, rank distance or rank correlation method.

    Args:
        method: 'accuracy', ..., 'levenshtein', ..., 'spearman', ...
        **kwargs: keyword arguments to pass to the metric function.
    """
    if method in METRIC_METHODS:
        return metric(a, b, method=method, **kwargs)
    elif method in DIST_METHODS:
        return dist(a, b, method=method, **kwargs)
    elif method in CORR_METHODS:
        return corr(a, b, method=method, **kwargs)
    else:
        raise Exception('Unknown method: {}'.format(method))

def balanced_accuracy(y_true, y_pred):
    """ Compute balanced accuracy between y_true and y_pred.

    Args:
        y_true: Ground truth in 2D dense format.
        y_pred: Predictions in 2D dense format.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    recip_y_true = 1 - y_true
    recip_y_pred = 1 - y_pred
    sensitivity = np.sum(y_true * y_pred, axis=0) / np.sum(y_true, axis=0)
    specificity = np.sum(recip_y_true * recip_y_pred, axis=0) / np.sum(recip_y_true, axis=0)
    balanced_acc = np.mean((sensitivity + specificity) / 2.)
    return balanced_acc

def accuracy_multilabel(y_true,y_pred):
    """ Soft multi-label accuracy.

    Args:
        y_true: Ground truth in 2D dense format.
        y_pred: Predictions in 2D dense format.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape)==1:
        x = np.where(y_true-y_pred == 0)[0]
        return(len(x) / y_true.shape[0])
    inter = np.sum(y_true * y_pred, axis=1)
    union = np.sum(np.maximum(y_true,y_pred), axis=1)
    return np.mean(inter / union)

def loss(x, y, method='absolute'):
    """ Compute the error between two scalars or vectors.

        Args:
            x: float, usually representing a ground truth value.
            y: float, usually representing a single prediction.
            method: 'absolute', 'squared', ...
    """
    # TODO
    if method == 'absolute':
        return np.abs(x - y)
    elif method == 'squared':
        return (x - y) ** 2
    else:
        raise Exception('Unknown method: {}'.format(method))

def relative_metric(y_true, y_pred_list, loss='absolute', ranking_function=None, **kwargs):
    """ ...

    For example you can compute the Mean Rank of Absolute Error (averaged by class)
    by calling relative_metrics(y_true, y_pred_list, loss='absolute', ranking_function=rk.borda, reverse=True)

    Args:
        y_true: Ground truth (format?)
        y_pred_list: List of predictions (format?)
        loss: 'method' argument to be passed to the function 'loss'
        ranking_function: Ranking method from rk.ranking. rk.score by default.
        **kwargs: Arguments to be passed to the ranking function.
    """
    # TODO
    if ranking_function is None:
        ranking_function = rk.score
    m = [loss(y_pred, y_true, method=loss).mean(axis=1) for y_pred in m_pred]
    m = pd.DataFrame(np.array(m), index=None)
    return ranking_function(m, reverse=True, **kwargs)

def metric(y_true, y_pred, method='accuracy', reverse_loss=False, missing_score=-1, unilabel=False):
    """ Compute a classification scoring metric between y_true and y_pred.

    Predictions format:
    [[0.2, 0.3, 0.5]
    [0.1, 0.8, 0.1]]
    ...

    Ground truth format:
    [[0, 0, 1]
    [0, 1, 0]]

    If y_true and y_pred are 1D they'll be converted using `to_dense` function.

    Args:
        y_true: Ground truth (format?)
        y_pred: Predictions (format?)
        method: Name of the metric. Metrics available: 'accuracy', 'balanced_accuracy', 'balanced_accuracy_sklearn', 'precision', 'average_precision', 'brier', 'f1_score', 'mxe', 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse', 'mae'
        reverse_loss: If True, return (1 - score).
        missing_score: (DEPRECATED) Value to return if the computation fails.
        unilabel: If True, only one label per example. If False it's multi-label case.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise Exception('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))
    if len(y_true.shape) == 1:
        y_true, y_pred = to_dense(y_true), to_dense(y_pred)
    # TODO: Lift, BEP (precision/recall break-even point), Probability Calibration, Average recall, (SAR)
    # PARAMETERS
    average = 'binary'
    if y_true.shape[1] > 2: # target is not binary
        average = 'micro'
    # PREPROCESSING
    if method in ['accuracy', 'balanced_accuracy', 'balanced_accuracy_sklearn', 'precision', 'f1_score', 'recall', 'jaccard']: # binarize with 0.5 threshold
        y_true, y_pred = to_binary(y_true, unilabel=unilabel, at_least_one_class=True), to_binary(y_pred, unilabel=unilabel, at_least_one_class=True)
    if method in ['balanced_accuracy_sklearn'] or average == 'binary': # sparse format
        y_true, y_pred = to_sparse(y_true), to_sparse(y_pred)

    # TODO
    #y_true, y_pred = np.array(y_true), np.array(y_pred)
    #average = 'binary'
    #if y_true.shape != y_pred.shape:
    #    raise Exception('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))
    #if len(y_true.shape) == 1:
    #    try:
    #        y_true, y_pred = to_dense(y_true), to_dense(y_pred) # TODO: problem here, e.g. [0.2, 0.8] vs [0, 1]
    #        if y_true.shape[1] > 2: # target is not binary
    #            average = 'micro'
    #    except: # TODO, TMP Ad-Hoc fix?
    #        y_true, y_pred = y_true.T, y_pred.T
    # TODO: Lift, BEP (precision/recall break-even point), Probability Calibration, Average recall, (SAR)
    # PREPROCESSING
    #if method in ['accuracy', 'balanced_accuracy', 'balanced_accuracy_sklearn', 'precision', 'f1_score', 'recall', 'jaccard']: # binarize with 0.5 threshold
    #    y_true, y_pred = to_binary(y_true, unilabel=unilabel, at_least_one_class=True), to_binary(y_pred, unilabel=unilabel, at_least_one_class=True)
    #if method in ['balanced_accuracy_sklearn'] or average == 'binary': # sparse format
    #    try:
    #        y_true, y_pred = to_sparse(y_true), to_sparse(y_pred)
    #    except:
    #        pass
    #        # TMP TODO
    #try:
    # COMPUTE SCORE
    if method == 'accuracy':
        score = accuracy_score(y_true, y_pred)
    elif method == 'balanced_accuracy':
        score = balanced_accuracy(y_true, y_pred)
    elif method == 'balanced_accuracy_sklearn':
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
    elif method == 'mae':
        score = mean_absolute_error(y_true, y_pred)
    elif method == 'sar':
        score = combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rmse'], method='mean')
    else:
        raise Exception('Unknown method: {}'.format(method))
    #except Exception as e: # could not compute the score
    #    print('Could not compute {}. Retuning missing_score. Error: {}'.format(method, e))
    #    return missing_score # MISSING SCORE
    # REVERSE LOSS
    is_loss = method in ['mxe', 'mse', 'rmse']
    if reverse_loss and is_loss:
        score = 1 - score
    return score

def combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rmse'], method='mean'):
    """ Combine several metrics as one.

    For example, you can compute SAR metric by calling:
    combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rms'])

    Args:
        metrics: List of metric names.
        ranking: A ranking system from ranky.
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

    Between 0 and 1+

    Args:
        method: 'hamming', 'levenshtein', 'kendall', 'winner', 'euclidean', 'winner_mistake', 'winner_distance', 'asymmetrical_winner_distance'.
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
    elif method in ['kendall', 'kendalltau']: # Absolute Kendall distance, defined below
        d = kendall_tau_distance(r1, r2)
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
    elif method == 'winner_distance':
        d = winner_distance(r1, r2)
    elif method == 'symmetrical_winner_distance':
        d = symmetrical_winner_distance(r1, r2)
    else:
        raise(Exception('Unknown distance method: {}'.format(method)))
    return d

def corr(r1, r2, method='swap', return_p_value=False):
    """ Levenshtein/Wasserstein type correlation between two ordinal distributions.

    Between -1 and 1.

    Args:
        method: 'swap', 'spearman', 'pearson'
        p_value: If True, return a tuple (score, p_value)
    """
    if method in ['swap', 'kendalltau']: # Kendalltau: swap distance
        c, p_value = kendalltau(r1, r2)
    elif method in ['spearman', 'spearmanr']: # Spearman rank-order
        c, p_value = spearmanr(r1, r2)
    elif method in ['pearson', 'pearsonr']: # Pearson correlation
        c, p_value = pearsonr(r1, r2)
    # Add weightedtau
    # Add weightedspearman
    else:
        raise(Exception('Unknown correlation method: {}'.format(method)))
    if return_p_value:
        return c, p_value
    return c

def kendall_tau_distance(r1, r2, normalize=False):
    """ Compute the absolute Kendall distance between two ranks (array-like).

    This distance represents the minimal number of neighbors swaps needed to
    transform r1 into r2. Basically Kendall tau b without scaling between -1 and 1.

    https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    >>> kendall_tau_distance([0, 1, 2], [1, 2, 0])
    2

    >>> kendall_tau_distance([0, 1, 2], [0, 1, 2])
    0

    Ties management:
    >>> kendall_tau_distance([0, 1, 1, 1], [1, 1, 1, 0])
    4

    Args:
        normalize: If True, divide the results by the length of the lists.
    """
    n = len(r1)
    if len(r1) != len(r2):
        print("WARNING: r1 and r2 don't have the same length ({} != {})".format(len(r1), len(r2)))
    distance = corr(r1, r2, method='kendalltau') # scipy's Kendall tau b
    distance = (1 - distance) * n * (n-1) / 4 # convert from correlation coeff to distance
    if normalize:
        distance = distance / n
    return distance

def kendall_w(matrix, axis=0, ties=False):
    """ Kendall's W coefficient of concordance.

    See https://en.wikipedia.org/wiki/Kendall%27s_W for more information.

    Args:
        matrix: Preference matrix.
        axis: Axis of judges.
        ties: If True, apply the correction for ties
    """
    if ties:
        return kendall_w_ties(matrix, axis=axis)
    matrix = rk.rank(matrix, axis=1-axis) # compute on ranks
    m = matrix.shape[axis] # judges
    n = matrix.shape[1-axis] # candidates
    denominator = m**2 * (n**3 - n)
    rating_sums = np.sum(matrix, axis=axis)
    S = n * np.var(rating_sums)
    return 12 * S / denominator

def kendall_w_ties(matrix, axis=0):
    """ Kendall's W coefficient of concordance with correction for ties.

    The goal of this correction is to avoid having a lower score in the presence of ties in the rankings.

    Args:
        matrix: Preference matrix.
        axis: Axis of judges.
    """
    if axis == 1:
        matrix = matrix.T
    m = matrix.shape[0] # judges
    n = matrix.shape[1] # candidates
    matrix = rk.rank(matrix, axis=1) # compute on ranks
    T = [] # correction factors, one by judge
    for j in range(m):
        _, counts = np.unique(matrix[j], return_counts=True) # tied groups
        correction = np.sum([(t**3 - t) for t in counts])
        T.append(correction)
    denominator = m**2 * n * (n**2 - 1) - m * np.sum(T)
    sum = np.sum([r**2 for r in np.sum(matrix, axis=0)])
    numerator = 12 * sum - 3 * m**2 * n * (n + 1)**2
    return numerator / denominator

def concordance(m, method='spearman', axis=0):
    """ Coefficient of concordance between ballots.

    This is a measure of agreement between raters.
    The computation is the mean of the correlation between all possible pairs of judges.

    Args:
        axis: Axis of raters.
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

def distance_matrix(m, method='spearman', axis=0, names=None, **kwargs):
    """ Compute all pairwise distances.

    Distances can be dist, corr, metric.

    Args:
        method: metric, distance or correlation to use.
        axis: axis of items to compare (0 for rows or 1 for columns).
        names: list of size m[axis] of names of objects to compare.
                      Will be overwritten by index or columns if m is a pd.DataFrame.
        **kwargs: keywords argument for the metric function.
    """
    dataframe = False
    if rk.is_dataframe(m):
        dataframe = True
        if axis == 0:
            names = m.index
        elif axis == 1:
            names = m.columns
        else:
            raise Exception('axis must be 0 or 1.')
        m = np.array(m)
    n = m.shape[axis]
    idx = range(n)
    dist_matrix = np.zeros((n, n))
    for pair in it.product(idx, repeat=2):
        i, j = pair[0], pair[1]
        r1 = np.take(m, i, axis=axis)
        r2 = np.take(m, j, axis=axis)
        d = any_metric(r1, r2, method=method, **kwargs)
        dist_matrix[i, j] = d
    if dataframe: # if m was originally a pd.DataFrame
        dist_matrix = pd.DataFrame(dist_matrix)
        if names is not None:
            dist_matrix.columns = names
            dist_matrix.index = names
    return dist_matrix

def auc_step(X, Y):
    """ Compute area under curve using step function (in 'post' mode).

    X: List of timestamps of size n
    Y: List of scores of size n
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

def get_valid_columns(solution):
    """ Get a list of column indices for which the column has more than one class.

    This is necessary when computing BAC or AUC which involves true positive and
    true negative in the denominator. When some class is missing, these scores
    don't make sense (or you have to add an epsilon to remedy the situation).

    Args:
        solution: array, a matrix of binary entries, of shape (num_examples, num_features)
    Returns:
        valid_columns: a list of indices for which the column has more than one class.
    """
    num_examples = solution.shape[0]
    col_sum = np.sum(solution, axis=0)
    valid_columns = np.where(1 - np.isclose(col_sum, 0) - np.isclose(col_sum, num_examples))[0]
    return valid_columns

#TODO
#def relative_consensus or consensus_graph

def winner_distance(r1, r2, reverse=False):
    """ Asymmetrical winner distance.

        This distance is the rank of the winner of r1 in r2, normalized by the number of candidates.
        (rank(r1 winner)) - 1 / (n - 1)
        Assuming no ties.

        Args:
          r1: 1D vector representing a judge.
          r2: 1D vector representing a judge.
          reverse: If True, lower is better.
    """
    r1, r2 = np.array(r1), np.array(r2)
    if reverse:
        w1 = np.argmin(r1) # r1 winner
    else:
        w1 = np.argmax(r1) # r1 winner
    return (rk.rank(r2)[w1] - 1) / (len(r2) - 1)

def symmetrical_winner_distance(r1, r2, reverse=False):
    """ Symmetrical winner distance.

        Average of dist(r1, r2) and dist(r2, r1).

        Args:
          r1: 1D vector representing a judge.
          r2: 1D vector representing a judge.
          reverse: If True, lower is better.
    """
    d1 = winner_distance(r1, r2, reverse=reverse)
    d2 = winner_distance(r2, r1, reverse=reverse)
    return (d1 + d2) / 2

def centrality(m, r, axis=0, method='swap'):
    """ Compute how good a ranking is by doing the sum of the correlations between the ranking and all ballots in m.

    Args:
        method: 'hamming', 'levenshtein' for distance. 'swap', 'spearman' for correlation.
    """
    if method in CORR_METHODS: # correlation
        scores = np.apply_along_axis(corr, axis, m, r, method) # best 1
    else: # distance
        scores = - np.apply_along_axis(dist, axis, m, r, method) # minus because higher is better, best 0
    return scores.mean()

def mean_distance(r, m, axis, method):
    """ Mean distance between r and all points in m.
    """
    return - centrality(m, r, axis=axis, method=method)

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
