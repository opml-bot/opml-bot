import numpy as np
import sympy.integrals.rubi.utility_function
from sympy import sympify
from sympy.utilities.lambdify import lambdastr
from typing import Callable
import autograd.numpy as npa
from copy import deepcopy


# preprocessing
def prepare_all(function: str,
                restriction: str,
                method: str) -> tuple:
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

    method: str
        Метод для решения задачи. В зависимости от него будут переписываться ограничения для задачи. Принимает одно из
        значений среди ['bnb', ...]

    Returns
    -------
    func: Callable
        Функция, представленная в виде питоновской функции.
    restr: list
        Список питоновских функций, которые представляют собой функции ограничений.
    """

    func = sympify(function)
    vars_ = set(func.free_symbols)
    func = to_callable(func)
    restriction = restriction.split(';')
    restr = []
    if method == 'bnb':
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

    return func, restr


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


if __name__ == '__main__':
    f = 'x1**2 + x2**2 + (0.5*1*x1 + 0.5*2*x2)**2 + (0.5*1*x1 + 0.5*2*x2)**4'
    subject_to = 'x1+x2<=0;2*x1-3*x2<=1'

    f, restr, p = prepare_all(f, subject_to, 'primal-dual', '')
    print(p)
    for i in restr:
        print(i(p) >= 0)