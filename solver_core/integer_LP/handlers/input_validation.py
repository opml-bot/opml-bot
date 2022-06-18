import re

from typing import Optional
from sympy.parsing.sympy_parser import parse_expr
from sympy import sympify, exp, Symbol

ALLOWED_OPERATIONS = ['log', 'ln', 'factorial', 'sin', 'cos', 'tan', 'cot', 'pi', 'exp', 'sqrt', 'root', 'abs']


def check_expression(expression: str) -> str:
    """
    Функция для проверки выражения на корректность. Принимает на вход строку с функцией
    в аналитическом виде, возвращает строку. Функция обязательно должна быть
    только от аргументов вида x1, x2, ..., xn.

    Parameters:
    ------------
    expression: str
        Строка содержащая функцию для проверки.

    Returns:
    -------
    str: str
        Функция в виде строки.
    """

    expression = expression.strip()
    if expression.find('—') != -1:
        expression = expression.replace('—', '-')

    if expression.find('–') != -1:
        expression = expression.replace('–', '-')
    checker = compile(expression, '<string>', 'eval')  # Может выдать SyntaxError, если выражение некорректно
    var_checker = re.compile(r'^x{1}[0-9]+$')

    for name in checker.co_names:
        if name not in ALLOWED_OPERATIONS:
            if not (var_checker.match(name) and name != 'x0'):
                raise NameError(f"The use of '{name}' is not allowed")

    function = sympify(expression, {'e': exp(1)}, convert_xor=True)
    return str(function)


def check_restr(restr_str: str, method: str, splitter: Optional[str] = ';') -> str:
    """
    Проверяет корректность и читаемость ограничений.

    Parameters
    ----------
    restr_str: str
        Строка с ограничениями в аналитическом виде.

    method: str
        Название метода для решения задачи.

    splitter: Optional[str] = ';'
        Строка-разделитель, которым разделены градиенты.

    Returns
    -------
    restrs: str
        Строка с ограничениями в аналитическом виде, разделенные ';'.
    """

    g = restr_str.split(splitter)

    ans = []
    for i in range(len(g)):
        if method == 'bnb':
            if g[i].find('<=') != -1 or g[i].find('>=') != -1:
                if g[i].count('=') > 1:
                    raise ValueError(f'Неправильно задано ограничение {g[i]}')
                if g[i].find('>=') != -1:
                    splitt = '>='
                elif g[i].find('<=') != -1:
                    splitt = '<='
                left, right = g[i].split(splitt)
                left, right = sympify(check_expression(left.strip())), sympify(check_expression(right.strip()))
                if splitt == '<=':
                    left -= right
                if splitt == '>=':
                    left = -left
                    right = -right
                    left -= right
                right -= right
                checked = str(left) + '<=' + str(right)
                ans.append(checked)
            else:
                raise ValueError(f'''Для метода {method} ограничения типа равенств пока
                 не поддерживаются, можем добавить.''')
                # if g[i].count('=') != 1:
                #     raise ValueError(f'Неправильно задано ограничение {g[i]}')
        if method == 'gomori':
            if g[i].count('=') > 1:
                raise ValueError(f'Неправильно задано ограничение {g[i]}')
            if g[i].find('>=') != -1:
                splitt = '>='
            elif g[i].find('<=') != -1:
                splitt = '<='
            elif g[i].find('=') != -1:
                splitt = '='
            left, right = g[i].split(splitt)
            left, right = sympify(check_expression(left.strip())), sympify(check_expression(right.strip()))
            left -= right
            d = dict(zip(list(left.free_symbols), [0]*len(left.free_symbols)))
            b = float(left.subs(d))
            left -= b
            strleft = f'{left}'
            for i in left.free_symbols:
                str_x = f'{i}'
                strleft = strleft.replace(str_x, str_x[:1] + '_' + str_x[1:])
            ans.append(strleft + f' {splitt.strip()} {int(b)}')
    restrs = ";".join(ans)
    return restrs


if __name__ == '__main__':
    func = 'x1**2 - x3'
    restr = 'x2 - x4 >= 3'
    meth = 'gomori'
    start = '0;4;0;0'

    # костяк проверок
    s = check_expression(func)
    print(s)
    r = check_restr(restr, method=meth)
    print(r)
