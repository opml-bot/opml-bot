import numpy as np
import re
import math

from math import sqrt
from math import isclose

import sympy.integrals.rubi.utility_function
from sympy import sympify, Symbol, simplify
from sympy.utilities.lambdify import lambdastr
from typing import Optional, Callable
import autograd.numpy as npa
from copy import deepcopy



try:
    from solver_core.inner_point.FirstPhase import FirstPhase
except ImportError:
    from ..FirstPhase import FirstPhase



# preprocessing
def prepare_all(function: str,
                restriction: str,
                method: str,
                started_point: Optional[str] = None,
                ds: Optional[float] = 1.) -> tuple:
    """
    Функция подготавливает входные данные. В случае некорректного формата или математически неправильных записей будет
    вызвана ошибка.

    Parameters
    ------------
    function: str
        Функция для оптимизации. Функция в аналитическом виде, записанная в виде строки.

    restriction: str
        Строка ограничений, разделенными точкой с запятой. Ограничения записываются в том же виде,
        что и функция: функция в аналитическом виде в строке.(ограничения вида (=, =>, <=).

    started_point: str
        Координаты стартовой точки. Должна быть внутренней!

    method: str
        Метод для решения задачи. В зависимости от него будут переписываться ограничения для задачи. Принимает одно из
        значений среди ['None', 'primal_dual', ...]

    Returns
    -------
    func: Callable
        Функция, представленная в виде питоновской функции.
    restr: list
        Список питоновских функций, которые представляют собой функции ограничений.
    point: np.ndarray
        Массив с координатами точки.
    """

    func = sympify(function)
    vars_ = set(func.free_symbols)
    func = to_callable(func)
    restriction = restriction.split(';')
    restr = []
    if method == 'Newton':
        for i in restriction:
            left, right = i.split('=')
            left = left.strip()
            right = right.strip()
            left, right = sympify(left), sympify(right)
            left -= right
            vars_ |= set(left.free_symbols)
            left = to_callable(left)
            restr.append(left)
    if method == 'primal-dual':
        for i in restriction:
            if i.find('>=') != -1:
                spliter = '>='
            elif i.find('<=') != -1:
                spliter = '<='
            left, right = i.split(spliter)
            left = left.strip()
            right = right.strip()
            left, right = sympify(left), sympify(right)
            vars_ |= set(left.free_symbols)
            left -= right
            left = to_callable(left)
            restr.append(left)
    if method == 'log_barrier':
        for i in restriction:
            if i.find('>=') != -1:
                spliter = '>='
            elif i.find('<=') != -1:
                spliter = '<='
            left, right = i.split(spliter)
            left = left.strip()
            right = right.strip()
            left, right = sympify(left), sympify(right)
            left -= right
            vars_ |= set(left.free_symbols)
            left = to_callable(left)
            restr.append(left)
    # начальная точка
    if started_point != '' and started_point != 'None':
        coords = started_point.split(';')
        point = []
        for i in range(len(coords)):
            point.append(float(coords[i].strip()))
        point = np.array(point)
    else:
        if method == 'Newton':
            raise ValueError
        n_vars = int(str(max(list(vars_), key=lambda x: int(str(x)[1:])))[1:])
        rewrited_restrs = []
        if method == 'primal-dual':
            for r in restr:
                func = deepcopy(r)
                rewrited = rewrite(func)
                rewrited_restrs.append(rewrited)
            restr = rewrited_restrs
        for i in range(5):
            try:
                point = FirstPhase(n_vars, restr, ds=ds).solve()
            except np.linalg.LinAlgError as e:

                print(f'{i+1} попытка найти начальную точку провалилась, попробуем запустить новую итерацию.')
            except (OverflowError, ValueError) as e:

                print(f'{i + 1} попытка найти начальную точку провалилась, попробуем запустить новую итерацию.')
            else:
                break
        else:
            raise ValueError('Не удалось найти начальную точку. Попробуйте запустить еще раз или проверьте совместность системы')

    return func, restr, point


def rewrite(restr: Callable) -> Callable:
    """
    Функция переделывает ограничения с >= на <=.

    Parameters
    ----------
    restr: list
        Список питоновских функций, которые являются ограничениями для задачи (вида g(x) >= 0).

    Returns
    -------
    new: list
        Список переписанных функций (вида g(x) <= 0).
    """
    r = deepcopy(restr)
    new = lambda x: -r(x)
    return new


def to_callable(expression: sympy.core) -> Callable:
    """
    Преобразует исходное выражение в функцию питона.

    Parameters
    ----------
    expression: sympy expression
        Преобразует выражение sympy в питоновскую функцию от массива.

    Returns
    -------
    func: Callable
        Питоновская функция от массива.
    """

    str_vars = [str(i) for i in expression.free_symbols]
    str_vars = sorted(str_vars, key=lambda x: int(x[1:]), reverse=True)
    dict_for_vars = {i: f'x[{int(i[1:]) - 1}]' for i in str_vars}
    func = lambdastr(['x'], expression)

    for i in str_vars:
        i = str(i)
        func = func.replace(i, dict_for_vars[i])
    # print(func[9:])
    func = 'f=' + func
    d = {}
    exec(func, {'math': npa}, d)
    func = d['f'] # готовая функция
    return func


def make_matrix(expressions: list, xs: set) -> tuple:
    """
    Функция преобразует линейные sympy выражения в матрицы.

    Parameters
    ----------
    expressions: list
        Список ограничений фунции (в виде списка строк).
    xs: set
        Множество переменных в задаче.

    Returns
    -------
    A: np.ndarray
        Матрица весов при x.
    b: np.ndarray
        Вектор весов справа.
    """

    A = []
    b = []
    for i in expressions:
        l, r = i.split('=')
        b.append(float(r))
        exp = sympify(l)
        d = dict(zip(xs, [0] * len(xs)))
        coefs = [0]*len(xs)
        for j in exp.free_symbols:
            d[j] = 1
            coefs[int(str(j)[1:]) - 1] = float(exp.subs(d))
            d[j] = 0
        A.append(coefs)
    A = np.array(A)
    b = np.array(b)
    return A, b



if __name__ == '__main__':
    f = 'x1**2 + x2**2 + (0.5*1*x1 + 0.5*2*x2)**2 + (0.5*1*x1 + 0.5*2*x2)**4'
    subject_to = 'x1+x2<=0;2*x1-3*x2<=1'

    f, restr, p = prepare_all(f, subject_to, 'primal-dual', '')
    print(p)
    for i in restr:
        print(i(p) >= 0)