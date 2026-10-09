import numpy as np
import pandas as pd
import pytest

import ranky as rk
from conftest import DATA_DIR


def read_task(name):
    return pd.read_csv(DATA_DIR / 'test_metric' / name, sep=' ', header=None)


class TestFormats:
    def test_dense_sparse(self):
        y = [0, 2, 1, 2]
        dense = rk.to_dense(y)
        assert dense.shape == (4, 3)
        np.testing.assert_array_equal(rk.to_sparse(dense), y)

    def test_bad_dimensions(self):
        with pytest.raises(ValueError):
            rk.to_dense([[0, 1]])
        with pytest.raises(ValueError):
            rk.to_sparse([0, 1])

    def test_to_binary(self):
        m = np.array([[0.2, 0.3, 0.1], [0.5, 0.6, 0.7]])
        np.testing.assert_array_equal(rk.to_binary(m), [[0, 0, 0], [0, 1, 1]])
        np.testing.assert_array_equal(rk.to_binary(m, unilabel=True), [[0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(rk.to_binary(m, unilabel=True, at_least_one_class=True), [[0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(rk.to_binary(m, at_least_one_class=True), [[0, 1, 0], [0, 1, 1]])
        np.testing.assert_array_equal(rk.to_binary(m, threshold=0.25), [[0, 1, 0], [1, 1, 1]])
        with pytest.raises(ValueError):
            rk.to_binary([0.2, 0.8], unilabel=True)


class TestMetric:
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    def test_values(self):
        assert rk.metric(self.y_true, self.y_pred, method='accuracy') == 0.75
        assert rk.metric(self.y_true, self.y_pred, method='mse') == 0.25
        assert rk.metric(self.y_true, self.y_pred, method='rmse') == 0.5
        assert rk.metric(self.y_true, self.y_pred, method='mae') == 0.25
        assert rk.metric(self.y_true, self.y_pred, method='precision') == 1
        assert rk.metric(self.y_true, self.y_pred, method='recall') == 0.5

    def test_reverse_loss(self):
        assert rk.metric(self.y_true, self.y_pred, method='rmse', reverse_loss=True) == 0.5
        assert rk.metric(self.y_true, self.y_pred, method='mae', reverse_loss=True) == 0.75
        assert rk.metric(self.y_true, self.y_pred, method='accuracy', reverse_loss=True) == 0.75

    def test_perfect(self):
        y = rk.to_dense([0, 1, 2, 2, 1])
        for method in ['accuracy', 'balanced_accuracy', 'f1_score', 'roc_auc', 'jaccard']:
            assert rk.metric(y, y, method=method) == pytest.approx(1), method
        assert rk.metric(y, y, method='rmse') == 0

    @pytest.mark.parametrize('method', [m for m in rk.METRIC_METHODS])
    @pytest.mark.parametrize('predictions', ['task.predict', 'task_proba.predict'])
    def test_all_methods_on_task(self, method, predictions):
        y_true, y_pred = read_task('task.solution'), read_task(predictions)
        if method == 'mxe' and predictions == 'task_proba.predict':
            pytest.skip('This file contains raw scores, not probabilities.')
        assert np.isfinite(rk.metric(y_true, y_pred, method=method))

    def test_errors(self):
        with pytest.raises(ValueError):
            rk.metric([0, 1], [0, 1, 1])
        with pytest.raises(ValueError):
            rk.metric([0, 1], [0, 1], method='unknown')

    def test_combined_metric(self):
        y_true, y_pred = read_task('task.solution'), read_task('task.predict')
        sar = rk.combined_metric(y_true, y_pred)
        assert 0 <= sar <= 1
        assert sar == pytest.approx(rk.metric(y_true, y_pred, method='sar'))
        assert rk.combined_metric(y_true, y_pred, metrics=['accuracy'], method='median') == rk.metric(y_true, y_pred)
        with pytest.raises(ValueError):
            rk.combined_metric(y_true, y_pred, method='max')


def test_balanced_accuracy():
    y_true = rk.to_dense([0, 0, 0, 1])
    y_pred = rk.to_dense([0, 0, 0, 0])
    assert rk.balanced_accuracy(y_true, y_pred) == 0.5


def test_accuracy_multilabel():
    assert rk.accuracy_multilabel([[1, 0, 1]], [[1, 0, 1]]) == 1
    assert rk.accuracy_multilabel([[1, 0, 1]], [[1, 1, 0]]) == pytest.approx(1 / 3)
    assert rk.accuracy_multilabel([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_loss():
    np.testing.assert_array_equal(rk.loss(np.array([1, 2]), np.array([2, 0])), [1, 2])
    np.testing.assert_array_equal(rk.loss(np.array([1, 2]), np.array([2, 0]), method='squared'), [1, 4])
    with pytest.raises(ValueError):
        rk.loss(1, 2, method='cubic')


class TestDist:
    def test_values(self):
        assert rk.dist([1, 2, 3], [1, 3, 2]) == pytest.approx(2 / 3)
        assert rk.dist([1, 2, 3], [1, 3, 2], method='levenshtein') == 2
        assert rk.dist([0, 1, 2], [1, 2, 0], method='kendall') == pytest.approx(2)
        assert rk.dist([0, 0], [3, 4], method='euclidean') == 5
        assert rk.dist([1, 2, 3], [2, 1, 3], method='winner') == 1
        assert rk.dist([1, 2, 3], [1, 3, 2], method='winner_mistake') == 0
        assert rk.dist([1, 2, 3], [2, 1, 3], method='winner_mistake') == 1

    def test_identical(self):
        r = [3, 1, 2, 4]
        for method in rk.DIST_METHODS:
            assert rk.dist(r, r, method=method) == 0, method

    @pytest.mark.parametrize('method', rk.DIST_METHODS)
    def test_any_metric_accepts_all_methods(self, method):
        assert np.isfinite(rk.any_metric([3, 1, 2, 4], [1, 3, 2, 4], method=method))

    def test_unknown(self):
        with pytest.raises(ValueError):
            rk.dist([1, 2], [2, 1], method='unknown')
        with pytest.raises(ValueError):
            rk.any_metric([1, 2], [2, 1], method='unknown')


class TestCorr:
    @pytest.mark.parametrize('method', rk.CORR_METHODS)
    def test_bounds(self, method):
        assert rk.corr([1, 2, 3, 4], [1, 2, 3, 4], method=method) == pytest.approx(1)
        assert rk.corr([1, 2, 3, 4], [4, 3, 2, 1], method=method) == pytest.approx(-1)
        assert rk.any_metric([1, 2, 3, 4], [4, 3, 2, 1], method=method) == pytest.approx(-1)

    def test_p_value(self):
        c, p = rk.corr([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], method='spearman', return_p_value=True)
        assert c == pytest.approx(1)
        assert 0 <= p < 0.05

    def test_unknown(self):
        with pytest.raises(ValueError):
            rk.corr([1, 2], [2, 1], method='unknown')


def test_kendall_tau_distance():
    assert rk.kendall_tau_distance([0, 1, 2], [1, 2, 0]) == pytest.approx(2)
    assert rk.kendall_tau_distance([0, 1, 2], [0, 1, 2]) == 0
    assert rk.kendall_tau_distance([0, 1, 2, 3], [0, 1, 3, 2], normalize=True) == pytest.approx(0.25)
    assert rk.kendall_tau_distance([0.2, 0.2, 0.2, 0.1], [0.1, 0.2, 0.2, 0.2]) == pytest.approx(4)
    with pytest.raises(ValueError):
        rk.kendall_tau_distance([0, 1, 2], [0, 1])


class TestKendallW:
    M2 = np.array([[1, 2.5, 2.5, 4], [1, 2.5, 2.5, 4], [1, 2.5, 2.5, 4]])

    def test_values(self):
        M3 = np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
        assert rk.kendall_w(self.M2) == pytest.approx(0.9)
        assert rk.kendall_w(self.M2, ties=True) == pytest.approx(1.0)
        assert rk.kendall_w(M3) == 0.0

    def test_perfect_agreement(self):
        m = np.array([[1, 2, 3, 4]] * 5)
        assert rk.kendall_w(m) == pytest.approx(1)
        assert rk.kendall_w(m.T, axis=1) == pytest.approx(1)

    def test_axis(self, M):
        assert rk.kendall_w(M, axis=1) == pytest.approx(rk.kendall_w(M.T, axis=0))
        assert rk.kendall_w(M, axis=1, ties=True) == pytest.approx(rk.kendall_w(M.T, axis=0, ties=True))

    def test_dataframe(self, M, df):
        assert rk.kendall_w(df) == pytest.approx(rk.kendall_w(M))
        assert rk.kendall_w(df, ties=True) == pytest.approx(rk.kendall_w(M, ties=True))
        assert rk.kendall_w(pd.DataFrame(self.M2), ties=True) == pytest.approx(1.0)


def test_concordance():
    m = np.array([[1, 2, 3, 4], [1, 2, 4, 3], [1, 2, 4, 3], [1, 3, 2, 4], [2, 1, 3, 4], [1, 4, 3, 2]])
    assert rk.concordance(np.array([[1, 2, 3, 4]] * 3)) == pytest.approx(1)
    assert -1 <= rk.concordance(m, axis=0) <= 1
    assert rk.concordance(m, axis=1) == pytest.approx(rk.concordance(m.T, axis=0))
    assert rk.concordance(pd.DataFrame(m)) == pytest.approx(rk.concordance(m))


class TestDistanceMatrix:
    def test_dataframe(self, template):
        d = rk.distance_matrix(template)
        assert isinstance(d, pd.DataFrame)
        assert list(d.index) == list(template.index) == list(d.columns)
        np.testing.assert_array_almost_equal(np.diag(d), 1) # spearman correlation
        np.testing.assert_array_almost_equal(d, d.T)
        d = rk.distance_matrix(template, axis=1, method='levenshtein')
        assert list(d.index) == list(template.columns)
        np.testing.assert_array_equal(np.diag(d), 0)

    def test_names(self, M):
        assert isinstance(rk.distance_matrix(M, method='euclidean'), np.ndarray)
        d = rk.distance_matrix(M, method='euclidean', names=list('abcde'))
        assert isinstance(d, pd.DataFrame)
        assert d.loc['d', 'e'] == pytest.approx(np.linalg.norm([0.2, 0.2, 0.2]))

    def test_bad_axis(self, M):
        with pytest.raises(ValueError):
            rk.distance_matrix(M, axis=2)


class TestWinnerDistance:
    def test_values(self):
        assert rk.winner_distance([1, 0.7, 0.2, 0.1, 0.1], [0.7, 1, 0.5, 0.4, 0.1]) == 0.25
        assert rk.winner_distance([1, 2, 3], [1, 2, 3]) == 0
        assert rk.winner_distance([1, 2, 3], [3, 2, 1]) == 1

    def test_reverse(self):
        assert rk.winner_distance([1, 2, 3], [1, 2, 3], reverse=True) == 0
        assert rk.winner_distance([1, 2, 3], [3, 2, 1], reverse=True) == 1

    def test_symmetrical(self):
        a, b = [1, 0.7, 0.2, 0.1, 0.1], [0.7, 1, 0.5, 0.4, 0.1]
        d = rk.symmetrical_winner_distance(a, b)
        assert d == rk.symmetrical_winner_distance(b, a)
        assert d == pytest.approx((rk.winner_distance(a, b) + rk.winner_distance(b, a)) / 2)


def test_centrality():
    m = np.array([[1, 1, 1], [3, 3, 3], [2, 2, 2]])
    r = [1, 3, 2]
    assert rk.centrality(m, r) == pytest.approx(1)
    assert rk.centrality(m, r, method='hamming') == 0
    assert rk.centrality(m, [3, 1, 2]) < 1
    assert rk.mean_distance(r, m, 0, 'euclidean') == 0


def test_auc_step():
    X, Y = [60], [0.5]
    t = np.log(2) / np.log(21) # transformed time
    assert rk.auc_step(X, Y) == pytest.approx((1 - t) * 0.5)
    assert X == [60] and Y == [0.5] # inputs are not modified
    with pytest.raises(ValueError):
        rk.auc_step([1, 2], [0.5])


def test_get_valid_columns():
    solution = np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(rk.get_valid_columns(solution), [1])


def test_correct_metric():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(X, y)
    assert rk.correct_metric(accuracy_score, model, X, y) == 1
    assert rk.correct_metric(roc_auc_score, model, X, y) == 1
