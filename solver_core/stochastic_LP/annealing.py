import numpy as np
from sympy import *
import math
import random
import plotly.express as px
import pandas as pd

from handlers.utils_annealing import restrict, calc_temp, transition_prob, do_transition


def annealing(function, restrictions, type_f='max', start_temp=30, num_iter=1000, plot_history=True):
    """
    Решение задачи оптимизации методом имитации отжига
    Parameters
    ----------
    function: str
        Функция в аналитическом виде
    restrictions: list
        Список ограничений
    type_f: str
        Максимум или минимум
    start_temp: int
        Начальная температура отжига
    num_iter: iter
        Количество итераций
    plot_history: boolean
        Строить ли график

    Returns
    -------
    x_current: list
        Точка оптимума
    E_current: int
        Значение функции в точки оптимума
    """

    coeffs, coef_rest, coef_f, signs = restrict(restrictions, function)

    flag = True
    while flag:
        x_current = []
        for i in range(len(coef_f)):
            x_current.append(random.randint(0, max(coef_rest)))
        flag_2 = True
        for k in range(len(coeffs)):
            value = 0
            for j in range(len(coeffs[k])):
                value += coeffs[k][j] * x_current[j]
            if signs[k] == 'le':
                if value > coef_rest[k]:
                    flag_2 = False
            elif signs[k] == 'me':
                if value < coef_rest[k]:
                    flag_2 = False
            elif signs[k] == 'm':
                if value <= coef_rest[k]:
                    flag_2 = False
            elif signs[k] == 'l':
                if value >= coef_rest[k]:
                    flag_2 = False
            elif signs[k] == 'e':
                if value != coef_rest[k]:
                    flag_2 = False
        if flag_2 == True:
            flag = False

    E_current = 0
    for i in range(len(x_current)):
        E_current += coef_f[i] * x_current[i]

    history = pd.DataFrame({'x': [0], 'y': [E_current]})

    for cur_iter in range(1, num_iter):

        flag = True
        while flag:
            x_candidate = []
            for i in range(len(coef_f)):
                x_candidate.append(random.randint(0, max(coef_rest)))
            flag_2 = True
            for k in range(len(coeffs)):
                value = 0
                for j in range(len(coeffs[k])):
                    value += coeffs[k][j] * x_candidate[j]
                if signs[k] == 'le':
                    if value > coef_rest[k]:
                        flag_2 = False
                elif signs[k] == 'me':
                    if value < coef_rest[k]:
                        flag_2 = False
                elif signs[k] == 'm':
                    if value <= coef_rest[k]:
                        flag_2 = False
                elif signs[k] == 'l':
                    if value >= coef_rest[k]:
                        flag_2 = False
                elif signs[k] == 'e':
                    if value != coef_rest[k]:
                        flag_2 = False
            if flag_2 == True:
                flag = False

        E_candidate = 0
        for i in range(len(x_candidate)):
            E_candidate += coef_f[i] * x_candidate[i]

        if type_f == 'max':

            if E_current < E_candidate:
                x_current = x_candidate
                E_current = E_candidate
            else:
                cur_iter_temp = calc_temp(start_temp, num_iter, cur_iter)

                diff_E = abs(E_candidate - E_current)

                cur_iter_transition_prob = transition_prob(diff_E, cur_iter_temp)

                make_transition = do_transition(cur_iter_transition_prob)

                if make_transition:
                    x_current = x_candidate
                    E_current = E_candidate

        if type_f == 'min':

            if E_current > E_candidate:
                x_current = x_candidate
                E_current = E_candidate
            else:
                cur_iter_temp = calc_temp(start_temp, num_iter, cur_iter)

                diff_E = abs(E_candidate - E_current)

                cur_iter_transition_prob = transition_prob(diff_E, cur_iter_temp)

                make_transition = do_transition(cur_iter_transition_prob)

                if make_transition:
                    x_current = x_candidate
                    E_current = E_candidate


    return x_current, E_current


if __name__ == '__main__':

    objective_function = '8x_1 + 6x_2'
    constraints = ['2x_1 + 5x_2 <= 19', '4x_1 + 1x_2 <= 16']
    x, f = annealing(objective_function, constraints)
    print(x, f)
