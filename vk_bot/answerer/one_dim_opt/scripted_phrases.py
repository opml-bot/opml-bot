from typing import NamedTuple


class Phrases(NamedTuple):
    OPT_ALG_SELECTION = 'Выбери алгоритм, с помощью которого хочешь выполнить одномерную оптимизацию: '
    INPUT_FUNC = 'Введи функцию, которую хочешь оптимизировать: \n\n' \
                 'Пример: x**2 + 0.5'
    INPUT_INTERVAL_X = 'Введи ограничение для отрезка: \n\n' \
                       'Пример: pi/2 pi/2'
    INPUT_OPTIONAL_PARAMS = 'Выбери необязательные параметры, которые хочешь задать: '
    INPUT_X0 = 'Введи начальную точку: \n\n' \
               'Пример: 12'
    INPUT_PRINT_INTERIM = 'Вывести промежуточные итерации?'
    ERROR = 'При обработке данных произошла ошибка: {}'


class Examples(NamedTuple):
    ACC = '10**-5'
    MAX_ITER = '125'
    MAX_ARG = '100'
    C1 = '10**-3'
    C2 = '10**-4'
