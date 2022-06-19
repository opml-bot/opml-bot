from typing import NamedTuple

from solver_core.search_for_extremes.handlers.operations_name_gen import allowed_operations


class Phrases(NamedTuple):
    TASK_TYPE_SELECTION = 'Давай найдем экстремум твоей функции.\n\n ' \
                          'Выбери тип задачи:'
    CLICK_BUTTON = 'Не понял... Нажми на клавиатуру ☝🏻'
    INPUT_VARS = 'Теперь тебе нужно ввести входные данные отдельными сообщениями.\n\n' \
                 'Начни с имен переменных. Они не могут начинаться с цифры,' \
                 'а также содержать что-то кроме латинских букв и цифр.\n\n' \
                 'Пример: x y'
    INPUT_FUNC = 'Отлично!\n\n' \
                 f'Теперь введи функцию. Доступные имена: {", ".join(allowed_operations)}\n\n' \
                 'Пример: x**2 + 0.5 * y**2'
    INPUT_RESTR = 'Все в порядке.\n\n' \
                   'Введи список ограничений.\n\n' \
                   'Пример: 2*x + 0.5*y <= 1; -3*x + 2*y >= 5; -6*x - 7*y = 5'
    INPUT_START_POINT = 'Готово.\n\n' \
                  'Знаешь ли ты начальную точку?'
    INPUT_EPS = 'Введи требующуюся точность.\n\n'\
                'Пример: 0.001'
    COMPUTE = 'Нажми кнопку и жди результата! :)'
    LINK = '\n\nПосмотреть на график -> https://opml-bot.herokuapp.com/'
    ERROR = 'При обработке данных произошла ошибка: {}'