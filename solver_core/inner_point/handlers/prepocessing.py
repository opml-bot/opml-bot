import numpy as np
import math

from math import sqrt

import sympy.integrals.rubi.utility_function
from sympy import sympify, Symbol, simplify
from sympy.utilities.lambdify import lambdastr
from typing import Optional, Callable


def prepare_func(func: str, variables: list, method: Optional[str] = 'None') -> Callable:
    """
    Преобразует функцию записанной в строковом виде в функицю питона, которая принимает на вход массив с координатами
    точки.

    Parameters:
    ------------
    func: str
        Функция в аналитическом виде, записанная в строке.

    variables: list
        Список из элементов типа sympy.Symbol. Представляют собой все переменные для функции.

    method: str
        Метод для решения задачи. В зависимости от него будут переписываться ограничения для задачи. Принимает одно из
        значений среди ['None', 'primal_dual', ...].

    Returns:
    -------
    function
        питоновская функция
    """

    vars = [str(i) for i in variables[::-1]]
    dict_for_channge = dict(zip(vars, [f'x[{int(i[1:]) - 1}]' for i in vars]))
    dict_for_methods = {'primal-dual': rewrite_for_primaldual, 'None': lambda x: x}
    func = sympify(func)
    func = dict_for_methods[method](func)
    vars_in_func = func.free_symbols
    func = lambdastr(['x'], func)
    for i in vars_in_func:
        i = str(i)
        func = func.replace(i, dict_for_channge[i])
    func = 'f=' + func
    d = {}
    if method == 'primal-dual':
        import autograd.numpy as npa
        exec(func, {'math': npa}, d)
    else:
        exec(func, {'math': math, 'sqrt': sqrt}, d)
    return d['f']


def get_variables(function: str) -> list:
    """
    Функция достает из записанной в аналитическом виде функции переменные.
    Замечание: так как по правилам ввода переменные должны иметь вид x1, x2, x3, ..., xn, то в случае если функция
    зависит только от переменных x1, x5 переменные x2, x3, x4 будут созданы автоматически

    Parameters
    ----------
    function: str
        Функция в аналитическом виде

    Returns
    -------
    variables: list
        Список с переменными типа sympy.Symbol, отсортированные по возрастанию индекса.
    """

    function = sympify(function)
    var = list(function.free_symbols)
    var_str = [str(i) for i in var]
    max_index = int(max(var_str, key=lambda x: x[1:])[1:])
    for i in range(1, max_index):
        if f'x{i}' not in var_str:
            var_str.append(f'x{i}')
    var_str.sort(key=lambda x: int(x[1:]))
    variables = [Symbol(i) for i in var_str]
    return variables


def rewrite_for_primaldual(function: sympy.core.relational):
    """
    Переписывает ограничения для решения задачи методом прямо-двойственной задачи. Все ограничения должны быть
    вида ***функция ограничения*** >= 0 (ограничения больше или равно).

    Parameters
    ----------
    function: str
         Функия ограниченияmethod

    Returns
    -------
    rewrited: sympy expression
        Переписання функция.
    """
    function = str(function)
    if function.find('<=') == -1 and function.find('>=') == -1 or function.count('<=') > 1 or function.count('>=') > 1:
        raise ValueError(f'Невозможно переписать ограничения {function} должным образом.')
    elif function.find('<=') != -1:
        left, right = function.split('<=')
        left = -sympify(left)
        right = -sympify(right)
        left = left - right
    elif function.find('>=') != -1:
        left, right = function.split('>=')
        left = sympify(left)
        right = sympify(right)
        left = left - right
    print(left, '>=',  right - right)
    rewrited = left
    return rewrited


def prepare_constraints(constrainst: list, variables: list, method: str) -> list:
    """
    Функция, которая подготавливает ограничения для дальнейшего решения задачи.

    Parameters
    ----------
    constrainst: list
        Список, содержащий функции ограничений (в виде строк).

    variables: list
        Список из элементов типа sympy.Symbol. Представляют собой все переменные для данной задачи.

    method: str
        Метод для решения задачи. В зависимости от него будут переписываться ограничения для задачи. Принимает одно из
        значений среди ['None', 'primal_dual', ...].

    Returns
    -------
    Ограничения в том виде, который нужен для того или иного метода.
    """
    if method == 'primal-dual':
        ans = []
        for i in constrainst:
            ans.append(prepare_func(i, variables, 'primal-dual'))
        return ans


if __name__ == '__main__':
    f = 'x1**2 + x2**2'
    consts = ['x1 >= 0', 'x1 - x2 <=100', 'x2<=0']
    vars = get_variables(f)
    a = prepare_func(f, vars)
    c = prepare_constraints(consts, vars, 'primal-dual')
    print(c)
