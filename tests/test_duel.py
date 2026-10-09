import numpy as np
import pytest

import ranky as rk

A = [0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5]
B = [0.1, 0.2, 0.3, 0.4, 0.4, 0.1, 0.2, 0.3, 0.4, 0.4]


def test_hard_wins():
    assert rk.hard_wins([1, 2, 3], [0, 1, 4])
    assert not rk.hard_wins([0, 1, 4], [1, 2, 3])
    assert not rk.hard_wins([1, 2], [1, 2])
    assert rk.hard_wins([0, 1, 4], [1, 2, 3], reverse=True)


def test_copeland_wins():
    assert rk.copeland_wins([1, 2, 3], [0, 1, 4]) == 1
    assert rk.copeland_wins([0, 1, 4], [1, 2, 3]) == 0
    assert rk.copeland_wins([1, 0], [0, 1]) == 0.5
    assert rk.copeland_wins([0, 1, 4], [1, 2, 3], reverse=True) == 1


def test_p_wins():
    assert rk.p_wins(A, B)
    assert not rk.p_wins(B, A)
    assert rk.p_wins(B, A, reverse=True)
    # 3 wins out of 3 is not significant, unless pval=1
    assert not rk.p_wins([1, 1, 1], [0, 0, 0])
    assert rk.p_wins([1, 1, 1], [0, 0, 0], pval=1)


def test_declare_ties():
    assert not rk.declare_ties(A, B)
    assert rk.declare_ties([1, 1, 1], [0, 0, 0])
    assert not rk.declare_ties([1, 1, 1], [0, 0, 0], pval=1)
    assert not rk.declare_ties([1, 1, 1], [0, 0, 0], comparison_func=rk.hard_wins)


def test_success_rate():
    assert rk.success_rate([1, 2, 3], [2, 2, 2]) == pytest.approx(1 / 3)
    assert rk.success_rate([1, 2, 3], [2, 2, 2], ties=True) == pytest.approx(0.5)
    assert rk.success_rate([1, 2, 3], [2, 2, 2], reverse=True) == pytest.approx(1 / 3)
    assert rk.success_rate(A, B) == 1


def test_relative_difference():
    assert rk.relative_difference([0, 0, 1], [0, 0, 1]) == 0
    assert rk.relative_difference([0, 0], [0, 0]) == 0
    assert rk.relative_difference([0.8, 0.1, 0.8], [0.2, 0.1, 0.2]) == pytest.approx(0.4)
    assert rk.relative_difference([0.8, 0.1, 0.8], [0.2, 0.1, 0.2], reverse=True) == pytest.approx(-0.4)


class TestBayes:
    def test_bayes_wins(self):
        a = [0, 0, 0.2, 0, 0.3]
        b = [1, 0.8, 1, 0.2, 0.2]
        assert not rk.bayes_wins(a, b)
        assert rk.bayes_wins(b, a)

    def test_bayes_score(self):
        p = rk.bayes_score(A, B)
        assert 0.5 < p <= 1
        assert p == rk.bayes_wins(A, B, score=True)

    def test_independant(self):
        np.random.seed(0)
        assert rk.bayes_wins(A, B, independant=True)


def test_pairwise_with_duel_functions(M):
    np.testing.assert_array_equal(rk.pairwise(M, wins=rk.hard_wins), [2, 4, 3, 1, 0])
    np.testing.assert_array_almost_equal(rk.pairwise(M, wins=rk.success_rate),
                                         [7 / 3, 4, 7 / 3, 4 / 3, 0])
