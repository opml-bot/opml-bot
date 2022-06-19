from typing import NamedTuple

from solver_core.search_for_extremes.handlers.operations_name_gen import allowed_operations

class Phrases(NamedTuple):
    TASK_TYPE_SELECTION = 'Давай построим регрессионную модель.\n\n ' \
                          'Если хочешь добавить регуляризацию, то жми кнопку "С регуляризацией"\n\n' \
                          'Выбери тип задачи:'
    CLICK_BUTTON = 'Не понял... Нажми на клавиатуру ☝🏻'
    INPUT_DATA = 'Теперь тебе нужно ввести входные данные.\n\n' \
                 'Присылай свой csv файл, где первый столбец - массив предикторов (X)' \
                 ', второй столбец - массив предсказываемой переменной (y)'
    COMPUTE = 'Нажми кнопку и жди результата! :)'
    RESULT = 'Полученная функция:'
    LINK = '\n\nПосмотреть на график -> https://opml-bot.herokuapp.com/'
    ERROR = 'При обработке данных произошла ошибка: {}'