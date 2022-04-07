import pytest
from funcs import funcs
from scipy.optimize import minimize # тут наши функции
import numpy as np

EPS = 0.0001

import sys
@pytest.mark.skipif(sys.version_info < (3,3),reason="requires python3.3")
def test_graient_const():
    for names in funcs.keys():
        flag_OK= sum(abs(minimize(funcs[names][0], funcs[names][2]).x - np.array(funcs[names][1]))) < EPS
        assert flag_OK
