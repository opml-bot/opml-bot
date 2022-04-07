import re

from typing import Optional
from sympy.parsing.sympy_parser import parse_expr
from sympy import sympify, exp, Symbol

ALLOWED_OPERATIONS = ['log', 'ln', 'factorial', 'sin', 'cos', 'tan', 'cot', 'pi', 'exp', 'sqrt', 'root', 'abs']


def check_expression(expression: str) -> tuple:
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

    variables: list
        Список переменных функции.
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
    try:
        max_index = max([int(str(i)[1:]) for i in list(function.free_symbols)])
        variables = [f'x{i}' for i in range(1, max_index+1)]
    except:
        variables = []
    return str(function), variables


def check_gradients(grad_str: str, var: list, splitter: Optional[str] = ';') -> str:
    """
    Проверяет корректность и читаемость градиентов, а также сверяет количество градиентов с количеством переменных.
    Если в качестве s передали 'False' или пустую строку, то никаких обработок в дальнейшем не будет, а градиент
    будет чситаться численно.

    Parameters
    ----------
    grad_str: str
        Строка с градиентами в аналитическом виде.

    var: list
        Список переменных.

    splitter: Optional[str] = ';'
        Строка-разделитель, которым разделены градиенты.

    Returns
    -------
    grads: str
        Строка с градиентами в аналитическом виде, разделенные ';'.
    """

    if grad_str == '' or grad_str == 'False':
        return grad_str
    if var:
        nvars = int(max(var, key=lambda x: int(x[1:]))[1:])
    else:
        nvars = 0

    g = grad_str.split(splitter)
    if len(g) < nvars:
        raise ValueError(f'Введено меньше градиентов чем переменных: {len(g)} < {nvars}')
    else:
        ans = []
        for i in range(len(g)):
            checked = check_expression(g[i])
            if checked[1]:
                nvars_in_grad = int(max(checked[1], key=lambda x: int(x[1:]))[1:])
            else:
                nvars_in_grad = 0
            if nvars_in_grad > nvars:
                raise ValueError('В градиенте больше переменных, чем в исходной функции')
            ans.append(checked[0])

    grads = ";".join(ans)
    return grads


def check_float(value: str) -> float:
    """
    Проверяет введеное значение на корректность и на наличие инъекций, а затем
    конвертирует в float, если это возможно. Поддерживает операции с pi и e.
    Parameters:
    ------------
    values: str
        строка в которой содержится выражение
    Returns:
    -------
    float
        значение переведенное из строки в float
    """
    value = value.strip()
    if value.find('^') != -1:
        value = value.replace('^', '**')
    checker = compile(value, '<string>', 'eval')  # Может выдать SyntaxError, если выражение некорректно
    for name in checker.co_names:
        if name not in ['pi', 'e', 'exp']:
            raise ValueError(f'Нельзя использовать имя {name}')
    value = float(parse_expr(value, {'e': exp(1)}))
    return value


def check_int(value: str) -> int:
    """
    Проверяет введеное значение на корректность и на наличие инъекций, а затем
    конвертирует в int, если это возможно.
    Parameters:
    ------------
    values: str
        строка в которой содержится выражение
    Returns:
    -------
    int
        значение переведенное из строки в int
    """
    if value.find('^') != -1:
        value = value.replace('^', '**')
    value = int(parse_expr(value))
    return value


def check_point(point_str: str, splitter: Optional[str] = ';') -> str:
    """
    Функция проверяет корректность введеной точки x0.

    Parameters
    ----------
    point_str: str
         Координаты точки в виде строки.

    splitter: Optional[str] = ';'
        Разделитель, которым разделены координаты в строке.

    Returns
    -------
     point: str
        Строка с координатами точки, разделенные знаком ';'.
    """

    coords = point_str.split(splitter)
    for i in range(len(coords)):
        coords[i] = str(check_float(coords[i]))
    points = ';'.join(coords)
    return points


def check_dimension(vars: list, started_point: str):
    """
    Проверяет, сходятся ли размерность функции с размерностью стартовой точкой, в случае если они не сошлись, кидает
    ошибку.
    Parameters
    ----------
    vars: list
        Список перменных.

    started_point: str
        Строка с координатами стартовой точки.
    """

    coord = started_point.split(';')
    if len(coord) != len(vars):
        raise ValueError('Размерности введеной точки и функции не сходятся')


if __name__ == '__main__':
    func = '(x1-2)**2 + (x3 - 4)**2 + x4'
    grad = '2*(x1-2);0; 2*(x3 - 4); 1'

    s = check_expression(func)
    print(s[0], s[1])
    print(check_gradients(grad_str=grad, var=s[1]))
    print(check_float('10^-5'))
