#from solver_core.inner_point.testing.funcs import funcs_str

import pandas as pd

df = pd.DataFrame(columns=['Название','Ньютон','Внутренняя точка','Прямо-двойственный','ответ'])
for i in funcs_str.keys():
    res = funcs_str[i]
    print(res)
    df['Название'] = i
    #df['Ньютон'] = Newton(res[0],res[1]['eq'],res[1])
