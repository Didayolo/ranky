from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


@pytest.fixture(autouse=True)
def no_show(monkeypatch):
    """ Do not open windows during tests, and close figures after each test. """
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)
    yield
    plt.close('all')


@pytest.fixture
def M():
    """ 5 candidates (rows) x 3 judges (columns). """
    return np.array([[0.3, 0.4, 0.6],
                     [0.8, 0.8, 0.8],
                     [0.1, 0.5, 0.7],
                     [0.2, 0.2, 0.2],
                     [0.0, 0.0, 0.0]])


@pytest.fixture
def df(M):
    return pd.DataFrame(M, index=['a', 'b', 'c', 'd', 'e'], columns=['j1', 'j2', 'j3'])


@pytest.fixture
def template():
    """ The preference matrix used in the README. """
    m = pd.read_csv(DATA_DIR / 'matrix.csv', index_col='index')
    return m.rename_axis(None, axis=0)
