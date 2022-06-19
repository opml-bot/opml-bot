
import numpy as np
from sympy import *
import math
import random
import matplotlib.pyplot as plt


def stochatic_gradient_descent(function,restrictions, m = 10, eps = 0.01,max_iter = 1000,eta = 0.01):
  '''
  Решение задачи оптимизации методом стохастического градиентного спуска
  Parameters
  ----------
  function: str
      Функция в аналитическом виде
  restrictions: list
      Список ограничений
  m: int
      Размер выборки
  eps: int
      Точность алгоритма
  max_iter: int
      Максимальное количество итераций
  eta: int
      Коэффициент шага

  Returns
  -------
  sample_x[pos]: list
      Точка оптимума
  f_x[pos]: int
      Значение функции в точки оптимума
  w_sample: list
      Значение весов
  '''


  coeffs, coef_rest, coef_f, signs = restrict(restrictions,function)

  sample_x = []

  for r in range(m):

    flag  = True
    while flag:
      x_current = [1]
      for i in range(len(coef_f)):
        x_current.append(random.randint(0,max(coef_rest)))
      flag_2 = True
      for k in range(len(coeffs)):
        value = 0
        for j in range(len(coeffs[k])):
          value += coeffs[k][j]*x_current[j+1]
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
    sample_x.append(x_current) 

  w_sample = []
  for i in range(len(x_current)):
    w_sample.append(random.random())
  
  f_x = []
  for j in sample_x:
    f_current = 0
    for i in range(len(j[1:])):
      f_current += coef_f[i]*j[1:][i]
    f_x.append(f_current)

  weight_dist = np.inf
  iter_num = 0
  
  while weight_dist > eps and iter_num < max_iter:
    pos = random.randint(0,len(sample_x)-1)

    loss_f = []
    for i in range(len(x_current)):
      loss_f.append((np.array(sample_x[pos]).dot(np.array(w_sample).reshape(-1,1))[0] - f_x[pos])*2*sample_x[pos][i]/m)

    w_new = []
    for i in range(len(loss_f)):
      w_new.append(w_sample[i] - loss_f[i]*eta)
    
    weight_dist = np.linalg.norm(np.array(w_sample) - np.array(w_new))
    w_sample = w_new

    iter_num+=1


  return sample_x[pos], f_x[pos],w_sample

if __name__ == '__main__':

    from solver_core.stochastic_LP.handlers.utils_sgd import *

    objective_function = '8x_1 + 6x_2'
    constraints = ['2x_1 + 5x_2 <= 19', '4x_1 + 1x_2 <= 16']
    x,f,w = stochatic_gradient_descent(objective_function,constraints)
    print(x,f,w)
