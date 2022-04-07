import pandas as pd

df = pd.DataFrame(columns=['Название', 'Алгоритм1'])
for k, names in enumerate(funcs.keys()):
    df.loc[k] = [f'{names}', f'{}']
    sum(abs(minimize(funcs[names][0], funcs[names][2]).x - np.array(funcs[names][1]))) < EPS
