import random

import numpy as np
import pandas as pd
import pytest

import ranky as rk


# 4 candidates (A, B, C, D) x 5 judges, used for multi-turn uninominal voting.
# Favorites: A, A, B, C, C. Second turn between A and C: A, A, C, C, C.
UNINOMINAL = np.array([[4, 4, 2, 2, 2],
                       [3, 2, 4, 3, 3],
                       [2, 3, 3, 4, 4],
                       [1, 1, 1, 1, 1]])


class TestRank:
    def test_matrix(self, M):
        expected = np.array([[2., 3., 3.], [1., 1., 1.], [4., 2., 2.], [3., 4., 4.], [5., 5., 5.]])
        np.testing.assert_array_equal(rk.rank(M), expected)

    def test_1d(self):
        np.testing.assert_array_equal(rk.rank([0.2, 0.9, 0.5]), [3, 1, 2])

    def test_ascending_and_reverse(self):
        np.testing.assert_array_equal(rk.rank([0.2, 0.9, 0.5], ascending=True), [1, 3, 2])
        np.testing.assert_array_equal(rk.rank([0.2, 0.9, 0.5], reverse=True), [1, 3, 2])
        np.testing.assert_array_equal(rk.rank([0.2, 0.9, 0.5], ascending=True, reverse=True), [3, 1, 2])

    def test_ties(self):
        np.testing.assert_array_equal(rk.rank([1, 1, 0]), [1.5, 1.5, 3])
        np.testing.assert_array_equal(rk.rank([1, 1, 0], method='min'), [1, 1, 3])

    def test_keeps_names(self, df):
        r = rk.rank(df)
        assert isinstance(r, pd.DataFrame)
        assert list(r.index) == list(df.index) and list(r.columns) == list(df.columns)
        s = rk.rank(df['j1'])
        assert isinstance(s, pd.Series)
        assert s['b'] == 1


class TestAverageRank:
    def test_values(self, M):
        expected = [2.66666667, 1., 2.66666667, 3.66666667, 5.]
        np.testing.assert_array_almost_equal(rk.average_rank(M), expected)
        np.testing.assert_array_almost_equal(rk.borda(M), expected)

    def test_axis(self):
        m = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5]])
        np.testing.assert_array_equal(rk.average_rank(m, axis=0), [2, 2, 2])
        np.testing.assert_array_equal(rk.average_rank(m, axis=1), [3, 2, 1])
        np.testing.assert_array_equal(rk.borda(m, axis=0), [2, 2, 2])
        np.testing.assert_array_equal(rk.borda(m, axis=1), [3, 2, 1])

    def test_median(self, M):
        np.testing.assert_array_equal(rk.average_rank(M, method='median'), [3, 1, 2, 4, 5])

    def test_reverse(self, M):
        np.testing.assert_array_almost_equal(rk.average_rank(M, reverse=True), 6 - rk.average_rank(M))
        np.testing.assert_array_almost_equal(rk.average_rank(M.tolist(), reverse=True), 6 - rk.average_rank(M))

    def test_dataframe(self, df):
        r = rk.average_rank(df)
        assert isinstance(r, pd.Series)
        assert r.idxmin() == 'b'

    def test_unknown_method(self, M):
        with pytest.raises(ValueError):
            rk.average_rank(M, method='mode')


def test_majority(M):
    np.testing.assert_array_equal(rk.majority(M), [0.4, 0.8, 0.5, 0.2, 0.])


def test_score(M, df):
    np.testing.assert_array_almost_equal(rk.score(M), [0.43333333, 0.8, 0.43333333, 0.2, 0.])
    np.testing.assert_array_almost_equal(rk.score(df.T, axis=0), rk.score(M))
    assert list(rk.score(df.T, axis=0).index) == list(df.index)


def test_dictator(M):
    r = rk.dictator(M)
    assert any(np.array_equal(r, M[:, j]) for j in range(M.shape[1]))


class TestUninominal:
    def test_single_turn(self, M):
        np.testing.assert_array_equal(rk.uninominal(M), [0, 3, 0, 0, 0])
        np.testing.assert_array_equal(rk.uninominal(UNINOMINAL), [2, 1, 2, 0])

    def test_two_turns(self):
        np.testing.assert_array_equal(rk.uninominal(UNINOMINAL, turns=2), [2, 1, 3, 0])
        np.testing.assert_array_equal(rk.uninominal(UNINOMINAL, turns=2, keep_ranking=False), [2, 0, 3, 0])

    def test_axis(self):
        np.testing.assert_array_equal(rk.uninominal(UNINOMINAL.T, axis=0, turns=2), [2, 1, 3, 0])

    def test_too_many_turns(self):
        with pytest.raises(ValueError):
            rk.uninominal(UNINOMINAL, turns=4)


class TestPairwise:
    def test_copeland(self, M):
        np.testing.assert_array_equal(rk.pairwise(M), [2., 4., 3., 1., 0.])
        np.testing.assert_array_equal(rk.copeland(M), [2., 4., 3., 1., 0.])

    def test_axis(self, M):
        np.testing.assert_array_equal(rk.pairwise(M.T, axis=0), [2., 4., 3., 1., 0.])

    def test_wins_kwargs(self, M):
        np.testing.assert_array_equal(rk.pairwise(M, wins=rk.p_wins, pval=0.2), [0., 0., 0., 0., 0.])

    def test_score(self, M):
        np.testing.assert_array_equal(rk.pairwise(M, score=True), np.array([2., 4., 3., 1., 0.]) / 4)

    def test_graph(self, M):
        r, graph = rk.pairwise(M, return_graph=True)
        assert graph.shape == (5, 5)
        np.testing.assert_array_equal(np.diag(graph), 0)
        np.testing.assert_array_equal((graph + graph.T)[~np.eye(5, dtype=bool)], 1)
        np.testing.assert_array_equal(graph.sum(axis=1), r)

    def test_dataframe(self, df):
        r = rk.pairwise(df)
        assert r.idxmax() == 'b'


def test_consensus():
    rank_M = np.array([[1., 2., 3.], [1., 3., 2.], [1., 2., 3.]])
    np.testing.assert_array_equal(rk.consensus(rank_M, axis=1), [True, False, False])
    np.testing.assert_array_equal(rk.consensus(rank_M.T, axis=0), [True, False, False])


def test_contains_ties():
    assert rk.contains_ties([0, 1, 2, 1, 2, 0])
    assert rk.contains_ties([0.4, 0.2, 0.2, 0.2])
    assert not rk.contains_ties([0, 1, 2, 10, -2, 4])
    assert not rk.contains_ties([0.4, 0.2, 0.1, 0.5])
    assert not rk.contains_ties([])


def test_weigher():
    assert rk.weigher(0) == 1
    np.testing.assert_array_equal(rk.weigher(np.array([0, 1, 3])), [1, 0.5, 0.25])
    with pytest.raises(ValueError):
        rk.weigher(0, method='linear')


class TestSelection:
    def test_select_k_best(self):
        assert list(rk.select_k_best([0.1, 0.9, 0.5], k=2)) == [1, 2]
        assert list(rk.select_k_best([0.1, 0.9, 0.5], k=2, reverse=True)) == [0, 2]
        assert rk.select_best(pd.Series([0.1, 0.9], index=['x', 'y'])) == 'y'

    @pytest.mark.parametrize('k', [0, -1, 4])
    def test_bad_k(self, k):
        with pytest.raises(ValueError):
            rk.select_k_best([0.1, 0.9, 0.5], k=k)

    def test_top_k_method(self):
        D = [0.9, 0.8, 0.1, 0.7]
        F = [0.1, 0.5, 0.9, 0.6]
        assert rk.top_k_method(D, F, k=2) == 1
        assert rk.top_k_method(D, F, k=4) == 2


class TestBootstrap:
    def test_shape(self, M, df):
        assert rk.bootstrap(M).shape == M.shape
        assert rk.bootstrap(M, axis=1, n=10).shape == (5, 10)
        assert isinstance(rk.bootstrap(df), pd.DataFrame)

    def test_holdout(self):
        np.random.seed(0)
        m = np.arange(20).reshape(20, 1)
        sample, holdout = rk.bootstrap(m, return_holdout=True)
        sample, holdout = set(sample.ravel()), set(holdout.ravel())
        assert sample.isdisjoint(holdout)
        assert sample | holdout == set(range(20))

    def test_joint(self):
        m = np.arange(10).reshape(10, 1)
        a, b = rk.joint_bootstrap([m, m * 10])
        np.testing.assert_array_equal(b, a * 10)


def test_to_series():
    assert isinstance(rk.to_series([1, 2]), pd.Series)
    assert len(rk.to_series(np.array([[1], [2], [3]]))) == 3
    s = rk.to_series(pd.DataFrame({'x': [1, 2]}, index=['a', 'b']))
    assert isinstance(s, pd.Series) and list(s.index) == ['a', 'b']


def test_random_swap():
    np.random.seed(0)
    r = np.arange(6)
    swapped = rk.random_swap(r, n=5, tie=0)
    assert sorted(swapped) == list(r)
    np.testing.assert_array_equal(r, np.arange(6)) # not modified in place


class TestOptimization:
    # Candidates in rows, two judges that agree: the best ranking is (2, 0, 1).
    m = np.array([[10, 10], [0, 0], [5, 5]])

    def test_brute_force(self):
        assert rk.brute_force(self.m, axis=0) == (2, 0, 1)

    def test_brute_force_distance(self):
        assert rk.brute_force(self.m, axis=0, method='euclidean') == (2, 0, 1)

    def test_evolution_strategy(self):
        np.random.seed(0)
        random.seed(0)
        m = np.array([[1, 1, 1], [4, 4, 4], [2, 2, 2], [3, 3, 3]])
        r, h = rk.evolution_strategy(m, axis=0, l=5, epochs=30, tie=0, history=True, verbose=True)
        assert rk.corr(r, m[:, 0]) == pytest.approx(1)
        assert len(h) == 30
        assert all(a <= b for a, b in zip(h, h[1:])) # the best ranking is never lost

    def test_center(self):
        m = np.array([[1, 1, 1], [4, 4, 4], [2, 2, 2], [3, 3, 3]])
        r = rk.center(m, method='euclidean', verbose=False, seed=0)
        np.testing.assert_array_almost_equal(r, [1, 4, 2, 3], decimal=2)

    def test_optimal_spearman_is_borda(self, template):
        """ Optimal rank aggregation with Spearman correlation is equivalent to Borda count. """
        borda_rank = rk.rank(rk.borda(template), reverse=True)
        optimal_spearman_rank = rk.rank(rk.center(template, method='spearman', verbose=False, seed=0))
        np.testing.assert_array_equal(borda_rank, optimal_spearman_rank)

    def test_kemeny_young(self):
        m = np.array([[1, 1, 2], [4, 4, 4], [2, 3, 1], [3, 2, 3]])
        r = rk.kemeny_young(m, verbose=False, seed=0, maxiter=50)
        np.testing.assert_array_equal(rk.rank(r), [4, 1, 3, 2])
