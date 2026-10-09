import numpy as np
import pytest

import ranky as rk

R = [5, 4, 3, 2, 1]


class TestGenerator:
    def test_sample_shape(self):
        g = rk.Generator().fit(R)
        assert g.sample(n=9).shape == (5, 9)
        np.testing.assert_array_equal(g.sample(), R)
        assert g.sample(return_single=False).shape == (5, 1)

    def test_fit_int(self):
        g = rk.Generator()
        g.fit(4)
        np.testing.assert_array_equal(g.sample(), [0, 1, 2, 3])

    @pytest.mark.parametrize('r', [1, [1]])
    def test_fit_too_short(self, r):
        with pytest.raises(ValueError):
            rk.Generator().fit(r)

    def test_not_fitted(self):
        with pytest.raises(RuntimeError):
            rk.Generator().sample()

    def test_swap_generator(self):
        np.random.seed(0)
        m = rk.SwapGenerator().fit(R).sample(n=20, N=3)
        assert m.shape == (5, 20)
        for j in range(m.shape[1]):
            assert sorted(m[:, j]) == sorted(R)
        # Judges are close to the reference ranking
        assert rk.corr(rk.score(m), R) > 0.5

    def test_gaussian_generator(self):
        np.random.seed(0)
        m = rk.GaussianGenerator().fit(R).sample(n=50, scale=0.1)
        assert m.shape == (5, 50)
        np.testing.assert_array_almost_equal(rk.score(m), R, decimal=1)

    def test_rankings_from_generated_judges(self):
        np.random.seed(0)
        m = rk.SwapGenerator().fit(R).sample(n=9)
        for r in [rk.score(m), rk.uninominal(m), rk.borda(m), rk.pairwise(m)]:
            assert len(r) == len(R)


class TestNoise:
    def test_neighbors_swap(self):
        np.random.seed(0)
        r = [1, 2, 3, 4]
        swapped = rk.neighbors_swap(r, N=1)
        assert r == [1, 2, 3, 4] # not modified in place
        assert rk.dist(r, swapped, method='kendall') == pytest.approx(1)

    @pytest.mark.parametrize('p', [0, -0.5, 1.5])
    def test_bad_probability(self, p):
        with pytest.raises(ValueError):
            rk.neighbors_swap([1, 2, 3], p=p)
        with pytest.raises(ValueError):
            rk.ranking_noise([1, 2, 3], p=p)

    def test_ranking_noise(self):
        np.random.seed(0)
        r = [1, 2, 3, 4, 5]
        assert sorted(rk.ranking_noise(r, n=10)) == r
        tied = rk.ranking_noise(r, method='tie', n=1)
        assert rk.contains_ties(tied)
        assert r == [1, 2, 3, 4, 5]
        with pytest.raises(ValueError):
            rk.ranking_noise(r, method='shuffle')

    def test_gaussian_noise(self):
        np.random.seed(0)
        noisy = rk.gaussian_noise([1, 2, 3], scale=0.01)
        assert noisy.shape == (3,)
        np.testing.assert_array_almost_equal(noisy, [1, 2, 3], decimal=1)
