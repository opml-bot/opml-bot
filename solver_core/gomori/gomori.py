# -*- coding: utf-8 -*-
"""gomori.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qkHSNTthNqjfHXVwCOlrDDOvF9GW0QZA
"""

from fractions import Fraction
from warnings import warn
import copy
import pandas as pd

from solver_core.gomori.utils import *

def gomory_solve(num_vars: int, constraints: list, objective_function: tuple):
    """
    Решите задачу целочисленного линейного программирования с помощью заданных ограничений и целевой функции.
    Изначально проверяется условие целочисленного значения решения, полученного симплексным методом,
    если решение целочисленное – оно выводится в ответ.
    На каждой итерации цикла проверяется условие целочисленной оптимальности целевой функции.
    Если это условие выполнено, задача считается решенной и итерации прекращаются, возвращается результирующее целочисленное оптимальное значение функции и целочисленный оптимальный план.
    В противном случае выполняется один шаг алгоритма метода плоскости среза (метод Гомори).:
    - формирование вырезки и добавление ее в текущую симплексную таблицу
    - выполнение одного шага симплексного метода
    num_vars: number of variables
    constraints: list of constraints
         (for example ['1x_1 + 2x_2 >= 4', '2x_3 + 3x_1 <= 5', 'x_3 + 3x_2 = 6'])
    objective_function: tuple in which two string values are specified: objective function
         (for example, '2x_1 + 4x_3 + 5x_2') optimization direction ('min' or 'max')
    return:
    integer_optimum (float): оптимальное целочисленное значение целевой функции, полученное методом Гомори
    integer_optimal_plane (list): оптимальный целочисленный план, полученный методом Гомори
    gomory_history (list): список, содержащий шаги преобразования симплексной таблицы
    basic_vars_history (list): список, содержащий шаги преобразования basic_vars
    """
    simplex_vals, simplex_solution, simplex_history, simplex_basic_vars_history = simplex_solve(
        num_vars,
        constraints,
        objective_function
    )
    simplex_table = list(simplex_history.values())[-1]
    basic_vars = list(simplex_basic_vars_history.values())[-1]

    gomory_history = simplex_history
    basic_vars_history = simplex_basic_vars_history
    if check_integer_condition(simplex_table):
        return simplex_table[0][-1], simplex_solution
    gomory_history[f'Initial Gomory-method'] = copy.deepcopy(simplex_table)
    basic_vars_history[f'Initial Gomory-method'] = copy.deepcopy(basic_vars)
    step = 1
    while not check_integer_condition(simplex_table):
        simplex_table, basic_vars = add_clipping(simplex_table, basic_vars)
        key_row = get_key_row(simplex_table)
        key_column = get_key_column(simplex_table, key_row)
        simplex_table, basic_vars = simplex_step(simplex_table, basic_vars, key_column, key_row)
        gomory_history[f'Gomory method step {step}'] = copy.deepcopy(simplex_table)
        basic_vars_history[f'Gomory method step {step}'] = copy.deepcopy(basic_vars)
        step += 1
    integer_optimum = simplex_table[0][-1]
    integer_optimal_plane = create_solution(simplex_table, basic_vars, num_vars)

    return integer_optimum, integer_optimal_plane, gomory_history, basic_vars_history

if __name__ == '__main__':
    objective_function = ('maximize', '8x_1 + 6x_2')
    constraints = ['2x_1 + 5x_2 <= 19', '4x_1 + 1x_2 <= 16']
    num_vars = 2
    gomory_vals, gomory_solution, gomory_history, basic_vars_history = gomory_solve(
        num_vars,
        constraints,
        objective_function
    )
    print('Gomory method steps:')
    print_history_table(gomory_history, basic_vars_history, gomory_solution)