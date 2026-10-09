#######################################
### EVALUATION, COMPARISON, METRICS ###
#######################################
# Scoring metrics, rank distances, rank correlations and agreement measures.

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein
from scipy.spatial.distance import hamming
from scipy.stats import kendalltau, spearmanr, pearsonr
import itertools as it
from sklearn.metrics import accuracy_score, balanced_accuracy_score, average_precision_score, f1_score, log_loss, precision_score, recall_score, jaccard_score, roc_auc_score, mean_squared_error, mean_absolute_error
from . import ranking as rk # circular import, only used at call time

METRIC_METHODS = ['accuracy', 'balanced_accuracy', 'balanced_accuracy_sklearn', 'precision', 'average_precision', 'f1_score', 'mxe', 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse', 'sar', 'mae']
CORR_METHODS = ['swap', 'kendalltau', 'spearman', 'spearmanr', 'pearson', 'pearsonr']
DIST_METHODS = ['hamming', 'levenshtein', 'kendall', 'winner', 'euclidean', 'winner_mistake', 'winner_distance', 'asymmetrical_winner_distance', 'symmetrical_winner_distance']

def arr_to_str(a):
    """ Concatenate the values of a 1D array into a string. """
    return "".join(str(x) for x in a)

def to_dense(y):
    """ Convert labels from sparse format (1D, class indices) to dense format (2D, one-hot).

    >>> to_dense([0, 2, 1])
    array([[1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    y = np.array(y)
    if len(y.shape) == 1:
        length = y.shape[0]
        dense = np.zeros((length, y.max()+1))
        dense[np.arange(length), y] = 1
        return dense
    else:
        raise ValueError('y must be 1-dimensional')

def to_sparse(y, axis=1):
    """ Convert labels from dense format (2D, one-hot or probabilities) to sparse format (1D, argmax).
    """
    y = np.array(y)
    if len(y.shape) == 2:
        sparse = np.argmax(y, axis=axis)
        return sparse
    else:
        raise ValueError('y must be 2-dimensional')

def to_binary(y, threshold=0.5, unilabel=False, at_least_one_class=False):
    """ Convert probabilities to binary values {0, 1}.

    If unilabel is False, values strictly greater than the threshold become 1 and the others 0.
    If unilabel is True, the argmax of each row becomes 1 and the others 0.

    Args:
        y: Vector or matrix to binarize. If unilabel or at_least_one_class is True,
            y must be a 2D matrix of probabilities.
        threshold: Threshold for binarization (0 if below or equal, 1 if strictly above).
        unilabel: If True, return only one 1 for each row.
        at_least_one_class: If True, the argmax of each row is set to 1 even if
            it is below the threshold.

    >>> to_binary([[0.2, 0.3, 0.1], [0.5, 0.6, 0.7]], at_least_one_class=True)
    array([[0, 1, 0],
           [0, 1, 1]])
    """
    y = np.array(y)
    if unilabel or at_least_one_class:
        if len(y.shape) != 2:
            raise ValueError('If unilabel is True or at_least_one_class is True, y must be in 2D dense probability format.')
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
    """ Compare a and b with any scoring metric, rank distance or rank correlation.

    Dispatch to `metric`, `dist` or `corr` depending on the method.

    Args:
        a: First array-like (ground truth or ranking).
        b: Second array-like (predictions or ranking).
        method: Name of the method, from METRIC_METHODS, DIST_METHODS or CORR_METHODS.
        **kwargs: Arguments passed to the underlying function.
    """
    if method in METRIC_METHODS:
        return metric(a, b, method=method, **kwargs)
    elif method in DIST_METHODS:
        return dist(a, b, method=method, **kwargs)
    elif method in CORR_METHODS:
        return corr(a, b, method=method, **kwargs)
    else:
        raise ValueError('Unknown method: {}'.format(method))

def balanced_accuracy(y_true, y_pred):
    """ Balanced accuracy, averaged over classes.

    For each class (column), compute the mean of sensitivity and specificity.

    Args:
        y_true: Ground truth in 2D dense binary format.
        y_pred: Predictions in 2D dense binary format.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    recip_y_true = 1 - y_true
    recip_y_pred = 1 - y_pred
    sensitivity = np.sum(y_true * y_pred, axis=0) / np.sum(y_true, axis=0)
    specificity = np.sum(recip_y_true * recip_y_pred, axis=0) / np.sum(recip_y_true, axis=0)
    balanced_acc = np.mean((sensitivity + specificity) / 2.)
    return balanced_acc

def accuracy_multilabel(y_true, y_pred):
    """ Soft multi-label accuracy (mean intersection over union of each row).

    Args:
        y_true: Ground truth in 2D dense format.
        y_pred: Predictions in 2D dense format.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) == 1:
        x = np.where(y_true-y_pred == 0)[0]
        return len(x) / y_true.shape[0]
    inter = np.sum(y_true * y_pred, axis=1)
    union = np.sum(np.maximum(y_true, y_pred), axis=1)
    return np.mean(inter / union)

def loss(x, y, method='absolute'):
    """ Compute the error between two scalars or vectors (element-wise).

    Args:
        x: Usually the ground truth.
        y: Usually the prediction.
        method: 'absolute' or 'squared'.
    """
    if method == 'absolute':
        return np.abs(x - y)
    elif method == 'squared':
        return (x - y) ** 2
    else:
        raise ValueError('Unknown method: {}'.format(method))

def metric(y_true, y_pred, method='accuracy', reverse_loss=False, missing_score=-1, unilabel=False):
    """ Compute a classification scoring metric between y_true and y_pred.

    Inputs are expected in dense format (one row per example, one column per class):

        y_true = [[0, 0, 1],
                  [0, 1, 0]]
        y_pred = [[0.2, 0.3, 0.5],
                  [0.1, 0.8, 0.1]]

    1D inputs are considered to be class indices and are converted with `to_dense`.

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        method: Name of the metric, one of: 'accuracy', 'balanced_accuracy',
            'balanced_accuracy_sklearn', 'precision', 'average_precision', 'f1_score',
            'mxe' (log loss), 'recall', 'jaccard', 'roc_auc', 'mse', 'rmse', 'mae',
            'sar' (mean of accuracy, roc_auc and 1 - rmse).
        reverse_loss: If True, losses ('mxe', 'mse', 'rmse', 'mae') are returned as (1 - loss).
        missing_score: Deprecated, ignored.
        unilabel: If True, there is only one label per example. If False, it is the multi-label case.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape. {} != {}'.format(y_true.shape, y_pred.shape))
    if len(y_true.shape) == 1:
        y_true, y_pred = to_dense(y_true), to_dense(y_pred)
    # TODO: Lift, BEP (precision/recall break-even point), Probability Calibration, Average recall
    # PARAMETERS
    average = 'binary'
    if y_true.shape[1] > 2: # target is not binary
        average = 'micro'
    # PREPROCESSING
    if method in ['accuracy', 'balanced_accuracy', 'balanced_accuracy_sklearn', 'precision', 'f1_score', 'recall', 'jaccard']: # binarize with 0.5 threshold
        y_true, y_pred = to_binary(y_true, unilabel=unilabel, at_least_one_class=True), to_binary(y_pred, unilabel=unilabel, at_least_one_class=True)
    if method in ['balanced_accuracy_sklearn'] or average == 'binary': # sparse format
        y_true, y_pred = to_sparse(y_true), to_sparse(y_pred)
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
        score = np.sqrt(mean_squared_error(y_true, y_pred))
    elif method == 'mae':
        score = mean_absolute_error(y_true, y_pred)
    elif method == 'sar':
        score = combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rmse'], method='mean')
    else:
        raise ValueError('Unknown method: {}'.format(method))
    # REVERSE LOSS
    is_loss = method in ['mxe', 'mse', 'rmse', 'mae']
    if reverse_loss and is_loss:
        score = 1 - score
    return score

def combined_metric(y_true, y_pred, metrics=['accuracy', 'roc_auc', 'rmse'], method='mean'):
    """ Combine several metrics into one.

    Losses are converted to (1 - loss) before being combined. The default
    corresponds to the SAR metric (squared error, accuracy, ROC AUC).

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        metrics: List of metric names (see `metric`).
        method: 'mean' or 'median'.
    """
    scores = [metric(y_true, y_pred, m, reverse_loss=True) for m in metrics]
    if method == 'mean':
        score = np.mean(scores)
    elif method == 'median':
        score = np.median(scores)
    else:
        raise ValueError('Unknown method: {}'.format(method))
    return score

def dist(r1, r2, method='hamming'):
    """ Distance between two ranked ballots.

    Available methods:

    - 'hamming': proportion of positions that differ.
    - 'levenshtein': edit distance between the ballots written as strings
      (values are concatenated, so it is only meaningful for single digit values).
    - 'kendall' (or 'kendalltau'): number of swaps of neighbors, see `kendall_tau_distance`.
    - 'winner': r2[i] - r1[i], where i is the position of the lowest value in r1.
    - 'euclidean': Euclidean distance.
    - 'winner_mistake': 0 if the lowest value is at the same position in both ballots, 1 otherwise.
    - 'winner_distance' (or 'asymmetrical_winner_distance'): see `winner_distance`.
    - 'symmetrical_winner_distance': see `symmetrical_winner_distance`.

    Args:
        r1: 1D array-like.
        r2: 1D array-like.
        method: Name of the method.
    """
    # Other possible distances:
    # https://math.stackexchange.com/questions/2492954/distance-between-two-permutations
    # https://people.revoledu.com/kardi/tutorial/Similarity/OrdinalVariables.html
    # Footrule, Damerau-Levenshtein, Cayley, Ulam, Chebyshev, Minkowski, Jaro-Winkler
    if method == 'hamming': # Hamming distance: number of differences
        d = hamming(r1, r2)
    elif method == 'levenshtein': # Levenshtein distance - deletion, insertion, substitution
        d = levenshtein(arr_to_str(r1), arr_to_str(r2))
    elif method in ['kendall', 'kendalltau']: # Absolute Kendall distance, defined below
        d = kendall_tau_distance(r1, r2)
    elif method == 'winner': # How much the ranked first in r1 is far from the first place in r2
        i = np.argmin(r1)
        d = r2[i] - r1[i]
    elif method == 'euclidean':
        d = np.linalg.norm(np.asarray(r1) - np.asarray(r2))
    elif method == 'winner_mistake': # 0 if the winner is the same (TODO: ties?)
        d = 1
        if np.argmin(r1) == np.argmin(r2):
            d = 0
    elif method in ['winner_distance', 'asymmetrical_winner_distance']:
        d = winner_distance(r1, r2)
    elif method == 'symmetrical_winner_distance':
        d = symmetrical_winner_distance(r1, r2)
    else:
        raise ValueError('Unknown distance method: {}'.format(method))
    return d

def corr(r1, r2, method='swap', return_p_value=False):
    """ Correlation between two ranked ballots, between -1 and 1.

    Args:
        r1: 1D array-like.
        r2: 1D array-like.
        method: 'swap' or 'kendalltau' (Kendall tau-b), 'spearman' or 'pearson'.
        return_p_value: If True, return a tuple (correlation, p_value).
    """
    if method in ['swap', 'kendalltau']: # Kendalltau: swap distance
        c, p_value = kendalltau(r1, r2)
    elif method in ['spearman', 'spearmanr']: # Spearman rank-order
        c, p_value = spearmanr(r1, r2)
    elif method in ['pearson', 'pearsonr']: # Pearson correlation
        c, p_value = pearsonr(r1, r2)
    # TODO: weightedtau, weighted spearman
    else:
        raise ValueError('Unknown correlation method: {}'.format(method))
    if return_p_value:
        return c, p_value
    return c

def kendall_tau_distance(r1, r2, normalize=False):
    """ Absolute Kendall distance between two rankings.

    This is the minimal number of swaps of neighbors needed to transform r1
    into r2. It is computed from Kendall's tau-b, so ties are supported.

    See https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient

    >>> float(kendall_tau_distance([0, 1, 2], [1, 2, 0]))
    2.0
    >>> float(kendall_tau_distance([0, 1, 1, 1], [1, 1, 1, 0])) # with ties
    4.0

    Args:
        r1: 1D array-like.
        r2: 1D array-like, of the same length as r1.
        normalize: If True, divide the result by the length of the rankings.
    """
    n = len(r1)
    if len(r1) != len(r2):
        raise ValueError("r1 and r2 must have the same length ({} != {})".format(len(r1), len(r2)))
    distance = corr(r1, r2, method='kendalltau') # scipy's Kendall tau b
    distance = (1 - distance) * n * (n-1) / 4 # convert from correlation coeff to distance
    if normalize:
        distance = distance / n
    return distance

def kendall_w(matrix, axis=0, ties=False):
    """ Kendall's W coefficient of concordance.

    Measures the agreement between judges, from 0 (no agreement) to 1 (complete agreement).
    See https://en.wikipedia.org/wiki/Kendall%27s_W

    Args:
        matrix: Preference matrix.
        axis: Axis of judges.
        ties: If True, apply the correction for ties.
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

    Without this correction, ties in the rankings lower the coefficient.

    Args:
        matrix: Preference matrix.
        axis: Axis of judges.
    """
    matrix = np.asarray(matrix)
    if axis == 1:
        matrix = matrix.T
    m = matrix.shape[0] # judges
    n = matrix.shape[1] # candidates
    matrix = rk.rank(matrix, axis=1) # compute on ranks
    T = [] # correction factors, one by judge
    for j in range(m):
        _, counts = np.unique(matrix[j], return_counts=True) # tied groups
        T.append(np.sum(counts**3 - counts))
    denominator = m**2 * n * (n**2 - 1) - m * np.sum(T)
    sum_squares = np.sum(np.sum(matrix, axis=0) ** 2)
    numerator = 12 * sum_squares - 3 * m**2 * n * (n + 1)**2
    return numerator / denominator

def concordance(m, method='spearman', axis=0):
    """ Mean correlation between all pairs of judges.

    This is a measure of agreement between judges.

    Args:
        m: Preference matrix.
        method: Correlation method (see `corr`).
        axis: Axis of judges.
    """
    m = np.asarray(m)
    idx = range(m.shape[axis])
    scores = []
    for i, j in it.combinations(idx, 2):
        r1 = np.take(m, i, axis=axis)
        r2 = np.take(m, j, axis=axis)
        scores.append(corr(r1, r2, method=method))
    return np.mean(scores)

def distance_matrix(m, method='spearman', axis=0, names=None, **kwargs):
    """ Compute all pairwise distances (or correlations, or scores) between rows or columns.

    Args:
        m: 2D matrix.
        method: Any method accepted by `any_metric`. Note that for a correlation
            (the default), higher means closer.
        axis: Axis of the items to compare (0 for rows or 1 for columns).
        names: Names of the items to compare. If m is a pd.DataFrame, its index
            or columns are used instead.
        **kwargs: Arguments passed to the metric function.

    Returns:
        A square matrix, as a pd.DataFrame if m is a pd.DataFrame or if names
        are given, as a np.ndarray otherwise.
    """
    if axis not in [0, 1]:
        raise ValueError('axis must be 0 or 1.')
    if rk.is_dataframe(m):
        names = m.index if axis == 0 else m.columns
        m = np.array(m)
    n = m.shape[axis]
    dist_matrix = np.zeros((n, n))
    for i, j in it.product(range(n), repeat=2):
        r1 = np.take(m, i, axis=axis)
        r2 = np.take(m, j, axis=axis)
        dist_matrix[i, j] = any_metric(r1, r2, method=method, **kwargs)
    if names is not None:
        dist_matrix = pd.DataFrame(dist_matrix, index=names, columns=names)
    return dist_matrix

def auc_step(X, Y):
    """ Area under a step curve ('post' mode), with time on a log scale.

    Used to evaluate learning curves (score as a function of time). Time is
    transformed with log(1 + t / 60) / log(1 + 1200 / 60), so 1200 is the
    end of the curve. The point (0, 0) is added at the beginning and the
    last score is extended to the end of the curve.

    Args:
        X: List of timestamps of size n.
        Y: List of scores of size n.
    """
    def transform_time(t, T=1200, t0=60):
        return np.log(1 + t / t0) / np.log(1 + T / t0)
    if len(X) != len(Y):
        raise ValueError("The length of X and Y should be equal but got " +
                         "{} and {} !".format(len(X), len(Y)))
    X = [0] + [transform_time(t) for t in X] + [1]
    Y = [0] + list(Y)
    Y.append(Y[-1])
    # Compute area
    area = 0
    for i in range(len(X) - 1):
        delta_X = X[i + 1] - X[i]
        area += delta_X * Y[i]
    return area

def get_valid_columns(solution):
    """ Get the indices of the columns containing more than one class.

    This is necessary when computing BAC or AUC which involves true positive and
    true negative in the denominator. When some class is missing, these scores
    don't make sense (or you have to add an epsilon to remedy the situation).

    Args:
        solution: Matrix of binary entries, of shape (num_examples, num_features).

    Returns:
        Array of indices of the valid columns.
    """
    num_examples = solution.shape[0]
    col_sum = np.sum(solution, axis=0)
    valid_columns = np.where(1 - np.isclose(col_sum, 0) - np.isclose(col_sum, num_examples))[0]
    return valid_columns

def winner_distance(r1, r2, reverse=False):
    """ Asymmetrical winner distance.

    The rank of the winner of r1 in r2, normalized to be between 0 and 1:
    (rank - 1) / (n - 1). Ties are not handled.

    Args:
        r1: 1D vector of scores representing a judge.
        r2: 1D vector of scores representing a judge.
        reverse: If True, lower is better.
    """
    r1, r2 = np.array(r1), np.array(r2)
    if reverse:
        w1 = np.argmin(r1) # r1 winner
    else:
        w1 = np.argmax(r1) # r1 winner
    return (rk.rank(r2, reverse=reverse)[w1] - 1) / (len(r2) - 1)

def symmetrical_winner_distance(r1, r2, reverse=False):
    """ Symmetrical winner distance.

    Average of winner_distance(r1, r2) and winner_distance(r2, r1).

    Args:
        r1: 1D vector of scores representing a judge.
        r2: 1D vector of scores representing a judge.
        reverse: If True, lower is better.
    """
    d1 = winner_distance(r1, r2, reverse=reverse)
    d2 = winner_distance(r2, r1, reverse=reverse)
    return (d1 + d2) / 2

def centrality(m, r, axis=0, method='swap'):
    """ How central a ranking is among the judges of m. Higher is better.

    This is the mean correlation between r and all the judges, or minus the
    mean distance if `method` is a distance.

    Args:
        m: Preference matrix.
        r: 1D ranking of the candidates.
        axis: Axis of candidates.
        method: A correlation method (see `corr`) or a distance method (see `dist`).
    """
    if method in CORR_METHODS: # correlation
        scores = np.apply_along_axis(corr, axis, m, r, method) # best 1
    else: # distance
        scores = - np.apply_along_axis(dist, axis, m, r, method) # minus because higher is better, best 0
    return scores.mean()

def mean_distance(r, m, axis, method):
    """ Opposite of `centrality`, used as the objective function by `rk.center`.
    """
    return - centrality(m, r, axis=axis, method=method)

def correct_metric(metric, model, X_test, y_test, average='weighted', multi_class='ovo'):
    """ Score a scikit-learn model on (X_test, y_test) with a scikit-learn metric.

    Use predicted probabilities if the model supports them, otherwise use
    predictions. Different call signatures are tried, so that most
    scikit-learn metrics work without extra configuration.

    Args:
        metric: A scikit-learn metric function, e.g. `sklearn.metrics.roc_auc_score`.
        model: A fitted model.
        X_test: Test data.
        y_test: Test labels.
        average: `average` argument of the metric, if it accepts it.
        multi_class: `multi_class` argument of the metric, if it accepts it.
    """
    try:
        y_pred = model.predict_proba(X_test) # SOFT
        try:
            score = metric(y_test, y_pred, average=average, multi_class=multi_class)
        except Exception:
            try:
                score = metric(y_test, y_pred, average=average)
            except Exception:
                score = metric(y_test, y_pred)
    except Exception:
        y_pred = model.predict(X_test) # HARD
        try:
            score = metric(y_test, y_pred, average=average, multi_class=multi_class)
        except Exception:
            try:
                score = metric(y_test, y_pred, average=average)
            except Exception:
                try:
                    score = metric(y_test, y_pred)
                except Exception:
                    labels = np.unique(y_pred)
                    score = metric(y_test, y_pred, labels=labels)
    return score
