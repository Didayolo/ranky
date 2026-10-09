import pytest

import ranky as rk
from conftest import DATA_DIR


def test_read_codalab_csv():
    m = rk.read_codalab_csv(DATA_DIR / 'chems.csv')
    assert m.index.name == 'User'
    assert list(m.columns) == ['< Rank >', 'Classification (AUC ROC)', 'Feature Selection (AUC ROC)']
    assert list(m.loc['chenj43']) == [1.0, 0.92, 0.96]
    assert all(m.dtypes == float)


@pytest.mark.parametrize('cell, expected', [
    ('0.78 (2)', 0.78),
    ('-1.00 (36)', -1.0),
    ('', ''),
    ('abc', 'abc'),
    (3, 3),
])
def test_str_to_float(cell, expected):
    assert rk.str_to_float(cell) == expected
