from solver_core.inner_point.testing.funcs import funcs_str


import pandas as pd

df = pd.DataFrame(columns=['Название','Метод Гамори','Метод ветвей и границ','Ответ'])
for i in funcs_str.keys():
    res = funcs_str[i]
    print(res)
    f = check_expression(res[0])
    subject_to = check_restr(res[1]['ineq'], method='bnb')
    # preprocessing
    f, subject_to = prepare_all(f, subject_to, 'bnb')
    df['Название'] = i
    df['Метод Гамори'] = gomory_solve(2,subject_to,['minimize',f])[0]
    df['Метод ветвей и границ'] = BNB(f,subject_to)
    df['Ответ'] = res[2]
