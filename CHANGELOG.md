# Changelog

## 1.0.0

### Fixed

- Compatibility with pandas >= 3 (`read_codalab_csv`) and scikit-learn >= 1.6 (`metric(..., method='rmse')`).
- `average_rank(m, method='median')` failed on numpy arrays, and `average_rank(m, reverse=True)` failed on lists.
- `evolution_strategy(..., verbose=True)` raised a `NameError`.
- `brute_force` returned a wrong result with distance methods (e.g. `method='euclidean'`).
- `winner_distance(..., reverse=True)` only reversed the first ranking.
- `kendall_w(..., ties=True)` failed on DataFrames.
- `any_metric` and `distance_matrix` rejected the 'kendall', 'winner_mistake' and 'symmetrical_winner_distance' methods; `dist` rejected 'asymmetrical_winner_distance'.
- `distance_matrix` ignored the `names` argument for numpy arrays.
- `auc_step` modified the list given as argument.
- `critical_difference` ignored its keyword arguments (e.g. `pval`).
- `show` misplaced the column names of heatmaps.
- `mds` used correlations as distances. Correlations are now converted with (1 - correlation).
- `scatterplot(..., dim=3)` crashed, and a 2D plot drawn after a 3D plot crashed.
- `tsne` failed with less than 31 points (perplexity is now adapted).
- `neighbors_swap`, `ranking_noise`, `tsne` and `mds` raised a `TypeError` or `NameError` instead of the intended error message.
- The probability `p` of `neighbors_swap`/`ranking_noise` and `tie` of `random_swap` were rounded (e.g. 0.3 was used as 1/3).
- `str_to_float('')` raised an `IndexError`.
- `select_k_best` accepted negative values of `k`.

### Changed

- Errors are raised as `ValueError` (or `RuntimeError` for an unfitted generator) instead of `Exception`. Code catching `Exception` still works.
- `Generator.fit` returns the generator, so calls can be chained: `rk.SwapGenerator().fit(r).sample(n=10)`.
- `kendall_tau_distance` raises a `ValueError` when the rankings have different lengths, instead of printing a warning.
- 'brier' was removed from `METRIC_METHODS`, as it was never implemented.
- Removed `relative_metric` (it could not run) and `tie` (an empty placeholder).
- Packaging moved from `setup.py` to `pyproject.toml`. Python >= 3.9 is required. The `python-Levenshtein` dependency is replaced by `Levenshtein`, its new name.
- Tests moved to `tests/` and run with pytest, with GitHub Actions instead of Travis CI.
- Docstrings and documentation improved.
