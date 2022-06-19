# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v4SaIj50eUCDmB1wawuaAYxotJftfFpp
"""

from fractions import Fraction
from warnings import warn
import copy
import pandas as pd

def construct_simplex_table(constraints, num_vars):
    """
    Строит исходную симплексную таблицу на основе заданных ограничений и целевой функции.

    """
    num_s_vars = 0  
    num_r_vars = 0  
    for expression in constraints:
        if '>=' in expression:
            num_s_vars += 1
        elif '<=' in expression:
            num_s_vars += 1
            num_r_vars += 1
        elif '=' in expression:
            num_r_vars += 1
    total_vars = num_vars + num_s_vars + num_r_vars

    coeff_matrix = [[Fraction("0/1") for _ in range(total_vars + 1)] for _ in range(len(constraints) + 1)]
    s_index = num_vars
    r_index = num_vars + num_s_vars
    r_rows = []  # stores the non -zero index of r
    for i in range(1, len(constraints) + 1):
        constraint = constraints[i - 1].split(' ')

        for j in range(len(constraint)):

            if '_' in constraint[j]:
                coeff, index = constraint[j].split('_')
                if constraint[j - 1] == '-':
                    coeff_matrix[i][int(index) - 1] = Fraction("-" + coeff[:-1] + "/1")
                else:
                    coeff_matrix[i][int(index) - 1] = Fraction(coeff[:-1] + "/1")

            elif constraint[j] == '<=':
                coeff_matrix[i][s_index] = Fraction("1/1")  # add surplus variable
                s_index += 1

            elif constraint[j] == '>=':
                coeff_matrix[i][s_index] = Fraction("-1/1")  # slack variable
                coeff_matrix[i][r_index] = Fraction("1/1")  # r variable
                s_index += 1
                r_index += 1
                r_rows.append(i)

            elif constraint[j] == '=':
                coeff_matrix[i][r_index] = Fraction("1/1")  # r variable
                r_index += 1
                r_rows.append(i)

            coeff_matrix[i][-1] = Fraction(constraint[-1] + "/1")
    return coeff_matrix, r_rows, num_s_vars, num_r_vars


def update_objective_function(simplex_table, objective_function, objective):
    objective_function_coeffs = objective_function.split()
    for i in range(len(objective_function_coeffs)):
        if '_' in objective_function_coeffs[i]:
            coeff, index = objective_function_coeffs[i].split('_')
            if objective_function_coeffs[i-1] == '-':
                simplex_table[0][int(index)-1] = Fraction(int(coeff[:-1]), 1)
            else:
                simplex_table[0][int(index)-1] = Fraction(-int(coeff[:-1]), 1)
            if 'max' in objective:
                simplex_table[0][int(index)-1] *= -1
    return simplex_table


def create_solution(simplex_table, basic_vars, num_vars):
    solution = {}
    for i, var in enumerate(basic_vars[1:]):
        if var < num_vars:
            solution['x_' + str(var + 1)] = simplex_table[i + 1][-1]
    for i in range(0, num_vars):
        if i not in basic_vars[1:]:
            solution['x_' + str(i + 1)] = 0
    return solution

def get_fraction(numerator, denominator):
    return Fraction(numerator, denominator)

def sum_rows(row1: list, row2: list):
    """
    Суммирует две строки текущей симплексной таблицы. Строки задаются в параметрах
    """
    row_sum = [0 for i in range(len(row1))]
    for i in range(len(row1)):
        row_sum[i] = row1[i] + row2[i]
    return row_sum


def multiply_const_row(const: float, row: list):
    """
    Умножает строку на константу.
    Константа и индекс умножаемой строки задаются в параметрах

    """
    mul_row = []
    for i in row:
        mul_row.append(const*i)
    return mul_row

def simplex_solve(num_vars: int, constraints: list, objective_function: tuple):
    """
    Решите задачу линейного программирования симплексным методом.
    В теле метода каждой итерации цикла проверяется условие оптимальности целевой функции
    при заданных ограничениях, если это условие верно, задача считается решенной и
    итерации прекращаются, в противном случае выполняется один шаг симплексного метода.
    constraints: list of constraints
        (for example ['1x_1 + 2x_2 >= 4', '2x_3 + 3x_1 <= 5', 'x_3 + 3x_2 = 6'])
    objective_function: tuple in which two string values are specified: objective function
        (for example, '2x_1 + 4x_3 + 5x_2') optimization direction ('min' or 'max')

    """
    simplex_table, r_rows, num_s_vars, num_r_vars = construct_simplex_table(constraints, num_vars)
    objective, objective_function = objective_function[0], objective_function[1]
    basic_vars = [0 for _ in range(len(simplex_table))]
    simplex_table, basic_vars, simplex_history = phase1(simplex_table, basic_vars, r_rows, num_vars, num_s_vars)
    r_index = num_r_vars + num_s_vars
    for i in basic_vars:
        if i > r_index:
            raise ValueError("Infeasible solution")
    simplex_table = delete_r_vars(simplex_table, num_vars, num_s_vars)
    simplex_table = update_objective_function(simplex_table, objective_function, objective)

    simplex_history = {}
    basic_vars_history = {}
    for row, column in enumerate(basic_vars[1:]):
        if simplex_table[0][column] != 0:
            const = -simplex_table[0][column]
            result = multiply_const_row(const, simplex_table[row])
            simplex_table[0] = sum_rows(simplex_table[0], result)
    simplex_history['Initial Simplex-method'] = copy.deepcopy(simplex_table)
    basic_vars_history['Initial Simplex-method'] = copy.deepcopy(basic_vars)
    step = 1
    while not check_condition(simplex_table):
        key_column = find_key_column(simplex_table)
        key_row = find_key_row(simplex_table, key_column)
        simplex_table, basic_vars = simplex_step(simplex_table, basic_vars, key_column, key_row)
        simplex_history[f'Simplex-method step {step}'] = copy.deepcopy(simplex_table)
        basic_vars_history[f'Simplex-method step {step}'] = copy.deepcopy(basic_vars)
        step += 1
    optimum = simplex_table[0][-1]
    optimal_plane = create_solution(simplex_table, basic_vars, num_vars)
    return optimum, optimal_plane, simplex_history, basic_vars_history


def check_condition(simplex_table: list):
    """
    Проверяет условие оптимальности целевой функции для текущей симплексной таблицы

    """
    F = simplex_table[0]
    negative_values = list(filter(lambda x: x <= 0, F))
    condition = len(negative_values) == len(F)
    return condition


def find_key_column(simplex_table: list):
    """
    Найдите разрешающий столбец в текущей симплексной таблице
    """
    key_columns = 0
    for i in range(0, len(simplex_table[0]) - 1):
        if abs(simplex_table[0][i]) >= abs(simplex_table[0][key_columns]):
            key_columns = i

    return key_columns


def find_key_row(simplex_table: list, key_column: int):
    """
    Найдите разрешающую строку в разрешающем столбце текущей симплексной таблицы.
    """
    min_val = float("inf")
    key_row = 0
    for i in range(1, len(simplex_table)):
        if simplex_table[i][key_column] > 0:
            val = simplex_table[i][-1] / simplex_table[i][key_column]
            if val < min_val:
                min_val = val
                key_row = i
    if min_val == float("inf"):
        raise ValueError("Unbounded solution")
    if min_val == 0:
        warn("Dengeneracy")
    return key_row


def simplex_step(simplex_table: list, basic_vars: list, key_column: int, key_row: int):
    """
    Выполняет один шаг симплексного алгоритма:
    - ввод новых базовых переменных и вывод старых переменных из базы,
    - поиск разрешающего элемента,
    - нормализация разрешающей строки к разрешающему элементу,
    - обнуление разрешающего столбца.
    """
    basic_vars[key_row] = key_column
    pivot = simplex_table[key_row][key_column]
    simplex_table = normalize_to_pivot(simplex_table, key_row, pivot)
    simplex_table = make_key_column_zero(simplex_table, key_column, key_row)
    return simplex_table, basic_vars


def normalize_to_pivot(simplex_table: list, key_row: int, pivot: int):
    """
    Делит разрешающую строку текущей симплексной таблицы на разрешающий элемент
    """
    for i in range(len(simplex_table[0])):
        simplex_table[key_row][i] /= pivot
    return simplex_table


def make_key_column_zero(simplex_table: list, key_column: int, key_row: int):
    """
    Обнуляет элементы разрешающего столбца текущей симплексной таблицы, за исключением элемента, стоящего в разрешающей строке методом Жордана-Гаусса
    """
    num_columns = len(simplex_table[0])
    for i in range(len(simplex_table)):
        if i != key_row:
            factor = simplex_table[i][key_column]
            for j in range(num_columns):
                simplex_table[i][j] -= simplex_table[key_row][j] * factor
    return simplex_table


def delete_r_vars(simplex_table: list, num_vars: int, num_s_vars: int):
    for i in range(len(simplex_table)):
        non_r_length = num_vars + num_s_vars + 1
        length = len(simplex_table[i])
        while length != non_r_length:
            del simplex_table[i][non_r_length-1]
            length -= 1
    return simplex_table


def phase1(simplex_table: list, basic_vars: list, r_rows: list, num_vars: int, num_s_vars: int):

    simplex_history = []
    r_index = num_vars + num_s_vars
    for i in range(r_index, len(simplex_table[0]) - 1):
        simplex_table[0][i] = -1
    for i in r_rows:
        simplex_table[0] = sum_rows(simplex_table[0], simplex_table[i])
        basic_vars[i] = r_index
        r_index += 1
    s_index = num_vars
    for i in range(1, len(basic_vars)):
        if basic_vars[i] == 0:
            basic_vars[i] = s_index
            s_index += 1
    while not check_condition(simplex_table):
        key_column = find_key_column()
        key_row = find_key_row(key_column=key_column)
        simplex_table, basic_vars = simplex_step(key_column, key_row)
        simplex_history.append(copy.deepcopy(simplex_table))
    return simplex_table, basic_vars, simplex_history

def check_integer_condition(simplex_table: list):
    """
    Проверяет условие целочисленной оптимальности целевой функции
    (все значения в строке целевой функции текущей симплексной таблицы отрицательны или равны 0 и являются целыми числами)
    simplex_table: current simplex table
    return: condition (bool): True or False, в зависимости от выполнения условия целочисленной оптимальности
    """
    B = [x[-1] for x in simplex_table[1:]]
    integer_values = list(filter(lambda x: x.denominator == 1, B))
    positive_values = list(filter(lambda x: x >= 0, B))
    condition1 = len(positive_values) == len(B)
    condition2 = len(integer_values) == len(B)
    condition = condition1 and condition2
    return condition


def find_max_fractional_index(simplex_table: list):
    """
    Выполняет поиск индекса строки текущей симплексной таблицы, в которой находится максимальное значение дробной части элемента
    simplex_table: current simplex table
    return: max_fractional_index (int): индекс строки, содержащей элемент с максимальной дробной частью
    """
    max_fractional_part_i = 1
    for i in range(1, len(simplex_table)):
        curr_fractional_part = abs(float(simplex_table[i][-1])) % 1
        max_fractional_part = abs(float(simplex_table[max_fractional_part_i][-1])) % 1
        if curr_fractional_part > max_fractional_part:
            max_fractional_part_i = i
    return max_fractional_part_i


def add_clipping(simplex_table: list, basic_vars: list):
    """
    Формирует новую отсечку метода Гомори:
    - поиск строки, содержащей элемент с максимальной дробной частью,
    - компиляция неравенства отсечения Гомори на основе найденной строки,
    - сведение результирующего неравенства к равенству путем добавления новой переменной,
    - введение новой переменной в основу и добавление отсечения к текущей симплексной таблице
    simplex_table: current simplex table
    basic_vars: current basic vars
    return:
    simplex_table: new simplex table
    basic_vars: new basic variables
    """
    index = find_max_fractional_index(simplex_table)
    simplex_table.append([0 for _ in simplex_table[0]])
    for i in range(len(simplex_table)):
        if i == len(simplex_table) - 1:
            simplex_table[i].insert(-1, 1)
        else:
            simplex_table[i].insert(-1, 0)
    for j, coeff in enumerate(simplex_table[index]):
        if coeff != 0:
            simplex_table[-1][j] = get_fraction(-(coeff.numerator % coeff.denominator), coeff.denominator)
    basic_vars.append(len(simplex_table) - 1)
    return simplex_table, basic_vars


def get_key_row(simplex_table: list):
    beta = [x[-1] if x[-1] < 0 else 0 for x in simplex_table[1:]]
    key_row = 0
    for b in range(len(beta)):
        if abs(beta[b]) >= abs(beta[key_row]):
            key_row = b
    return key_row + 1


def get_key_column(simplex_table: list, key_row: int):
    tetha = [x / y if y < 0 else float('inf') for x, y in
             zip(simplex_table[0][:-2], simplex_table[key_row][:-2])]
    key_column = 0
    for t in range(len(tetha)):
        if tetha[t] <= tetha[key_column]:
            key_column = t
    return key_column

def print_history_table(table_history, basic_vars_history, solution):
    """
    История метода печати
    table_history: history of changing simplex table
    basic_vars_history: list containing the steps of simplex table conversion
    solution: list containing the steps of basic_vars conversion
    return: None
    """
    for step, table in table_history.items():
        print(step)
        simplex_table = pd.DataFrame(
            table,
            columns=[f'x{i + 1}' if i < len(table[0]) - 1 else 'b' for i in range(len(table[0]))],
            index=[f'x{basic_vars_history[step][i] + 1}' if i != 0 else 'f(x)' for i in
                   range(len(basic_vars_history[step]))]
        )
        print(simplex_table)
        print()
    print('Optimal point:')
    for var, value in solution.items():
        print(var, value)
