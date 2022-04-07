import numpy as np
import math

from math import sqrt
from sympy import sympify, Symbol
from sympy.utilities.lambdify import lambdastr
from typing import Optional, Callable


def prepare_func(func: str, variables: list) -> Callable:
    """
    Преобразует функцию записанной в строковом виде в функицю питона, которая принимает на вход массив с координатами
    точки.

    Parameters:
    ------------
    func: str
        Функция в аналитическом виде, записанная в строке.

    variables: list
        Список из элементов типа sympy.Symbol. Представляют собой все переменные для функции.

    Returns:
    -------
    function
        питоновская функция
    """

    vars = [str(i) for i in variables[::-1]]
    dict_for_channge = dict(zip(vars, [f'x[{int(i[1:]) - 1}]' for i in vars]))
    func = sympify(func)
    vars_in_func = func.free_symbols
    func = lambdastr(['x'], func)
    for i in vars_in_func:
        i = str(i)
        func = func.replace(i, dict_for_channge[i])
    func = 'f=' + func
    d = {}
    exec(func, {'math': math, 'sqrt': sqrt}, d)
    return d['f']


def prepare_func_newton(func: str, variables: list) -> Callable:
    """
    Преобразует функцию записанной в строковом виде в функицю питона, которая принимает на вход массив с координатами
    точки.

    Parameters:
    ------------
    func: str
        Функция в аналитическом виде, записанная в строке.

    variables: list
        Список из элементов типа sympy.Symbol. Представляют собой все переменные для функции.

    Returns:
    -------
    function
        питоновская функция
    """
    import autograd.numpy as npa

    vars = [str(i) for i in variables[::-1]]
    dict_for_channge = dict(zip(vars, [f'x[{int(i[1:]) - 1}]' for i in vars]))
    func = sympify(func)
    vars_in_func = func.free_symbols
    func = lambdastr(['x'], func)
    for i in vars_in_func:
        i = str(i)
        func = func.replace(i, dict_for_channge[i])
    func = 'f=' + func
    d = {}
    exec(func, {'math': npa, 'sqrt': npa.sqrt, 'exp': npa}, d)
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


def prepare_gradient(s: str, variables: list) -> Callable:
    """
    Если градиенты заданы, то эта функция преобразует их в функцию от массива. Если не заданы, то она создает
    функцию от массива, которая численно считает градиент.

    Замечание: при численном вычислении градиента передается больше параметров в функцию, чем если бы градиенты были
    заданы. Для того чтобы обращения к функциям в классах для решения задач были одинаковы, в случае если градиенты
    заданы параметры function и delta_x тоже указаны, но на вычисления градиента никак не влияют.

    Parameters
    ----------
    s: str
        Строка с записанными функциями-градиентами, разделенные строкой splitter.

    variables: list
        Список переменных. Список состоит из значений типа sympy.Symbol.

    Returns
    -------
    f: Callable
        Функция градиента. Принимает исходную функцию, точку в которой надо считать градиент и delta_x для расчетов.
        Если градиенты заданы, то принимаются те же параметры, но для расчетов используется только точка.
    """

    gra = []
    if s:
        if s != 'False':
            grads = s.split(';')
            for func in grads:
                k = prepare_func(func=func, variables=variables)
                gra.append(k)
            def gradient(function: Callable, x0: np.ndarray, delta_x=1e-8) -> np.ndarray:
                ans = []
                for i in range(len(gra)):
                    ans.append(gra[i](x0))
                return np.array(ans)
            return gradient
    def gradient(function: Callable,
                 x0: np.ndarray,
                 delta_x=1e-8) -> np.ndarray:
        """
        Численно вычисляет градиент. Параметр  delta_x отвечает за шаг изменения аргумента в проивзводной.

        Parameters
        ----------
        function: Callable
            Функция от которой берут гралиент в смысле питоновской фунции.

        x0: np.ndarray
            Точка, в которой вычисляют градиент

        delta_x: Optional[float] = 1e-8
             Шаг для производной.

        Returns
        -------
        grad: np.ndarray
            Значения градиента в точке x0
        """

        grad = []
        for i in range(len(x0)):
            delta_x_vec_plus = x0.copy()
            delta_x_vec_minus = x0.copy()
            delta_x_vec_plus[i] += delta_x
            delta_x_vec_minus[i] -= delta_x
            grad_i = (function(delta_x_vec_plus) - function(delta_x_vec_minus)) / (2 * delta_x)
            grad.append(grad_i)

        grad = np.array(grad)
        return grad
    return gradient


def prepare_point(pont_str: str) -> np.ndarray:
    """
    Преобразует точку из записи в строчном виде, с координатами разделенными точкой с запятой, в массив NumPy.

    Parameters
    ----------
    pont_str: str
        Строка с координатами точки, разделитель - ';'

    Returns
    -------
    point: np.ndarray
        Массив с координатами точки.
    """

    coords = pont_str.split(';')
    point = []
    for i in range(len(coords)):
        point.append(float(coords[i]))
    point = np.array(point)
    return point


if __name__ == '__main__':
    func = 'x2**2 + x7 + x9 - 3'
    grads = ['0', '2*x2', '0', '0', '0', '0', '1', '0', '1']
    grads = ";".join(grads)
    print(grads)
    xs = get_variables(func)
    f = prepare_func(func, xs)
    point = ['2' for i in range(len(xs))]
    point = ';'.join(point)
    point = prepare_point(point)
    grads = prepare_gradient(grads, xs)
    print(grads(f, point))
    print(f(point))
