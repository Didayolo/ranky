#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Compute rankings in Python.

    import ranky as rk

All functions are available from the top-level `rk` namespace. They are
organized in the following modules:

- `ranky.ranking`: ranking systems (Borda count, Copeland, Kemeny-Young, ...).
- `ranky.metric`: scoring metrics, rank distances and correlations, agreement measures.
- `ranky.duel`: pairwise comparison functions and statistical tests.
- `ranky.visualization`: plots of rankings and preference matrices.
- `ranky.generator`: generation of synthetic judges.
- `ranky.utilities`: reading Codalab leaderboards.

A preference matrix has one row per candidate and one column per judge.
Most functions accept np.ndarray or pd.DataFrame; with a DataFrame, the
names of the candidates and judges are kept in the results.

Made by Adrien PAVAO, 2020.
"""

from .ranking import *
from .metric import *
from .duel import *
from .visualization import *
from .generator import *
from .utilities import *

__version__ = '1.0.0'
