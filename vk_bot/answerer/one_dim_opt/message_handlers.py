from vk_api.vk_api import VkApiMethod

from solver_core.one_dim_opt.brandt import Brandt
from solver_core.one_dim_opt.golden_ratio import GoldenRatio
from solver_core.one_dim_opt.handlers.input_validation import check_expression, check_limits, check_float, check_int
from solver_core.one_dim_opt.handlers.preprocessing import prepare_func, prepare_limits
from solver_core.one_dim_opt.parabola import Parabola
from vk_bot.answerer.one_dim_opt.keyboards import Keyboards
from vk_bot.answerer.one_dim_opt.onedimopt import OneDimOpt
from vk_bot.answerer.one_dim_opt.scripted_phrases import Phrases
from vk_bot.answerer.response_init import Response
from vk_bot.database import BotDatabase
from vk_bot.user import User


class Handlers:
    """
    Генератор ответов пользователю.

    Parameters
    ----------
    vk_api_method : VkApiMethod
        Объект соединения с VK и набор методов API.
    db : BotDatabase
        Объект для взаимодействия с базой данных.
    user : User
        Объект для взаимодействия с данными пользователя.
    """

    def __init__(self, vk_api_method: VkApiMethod, db: BotDatabase, user: User, onedimopt: OneDimOpt):
        self.vk = vk_api_method
        self.db = db
        self.user = user
        self.onedimopt = onedimopt
        self.response = Response(self.user.user_id)

    def opt_alg_selection(self) -> Response:
        """
        Выбор алгоритма для оптимизации.
        Шаг: opt_alg_selection

        Returns
        -------
        Response
            Сообщение для пользователя.
        """
        self.onedimopt.update_step('opt_alg_selection')
        self.response.set_text(Phrases.OPT_ALG_SELECTION)
        self.response.set_keyboard(Keyboards().for_opt_alg_selection())
        return self.response

    def input_func(self) -> Response:
        """
        Предложение ввести функцию для оптимизации.
        Шаг: input_func

        Returns
        -------
        Response
            Сообщение для пользователя.
        """
        self.onedimopt.update_step('input_func')
        self.response.set_text(Phrases.INPUT_FUNC)
        return self.response

    def func(self, text: str, task_type: str) -> Response:
        """
        Обработка введенной функции. Предложение ввести следующие данные.
        Шаг:

        Parameters
        ----------
        text : str
            Текст сообщения, отправленного пользователем.
        task_type : str
            Тип алгоритма.

        Returns
        -------
        Response
            Сообщение для пользователя.
        """

        try:
            func = check_expression(text)
            self.onedimopt.update_func(func)
            if task_type != "bfgs":
                self.response.set_text(Phrases.INPUT_INTERVAL_X)
                self.onedimopt.update_step('input_interval_x')
            else:
                self.response.set_text(Phrases.INPUT_X0)
                self.onedimopt.update_step('input_x0')
            return self.response
        except Exception as e:
            return self.error(e)

    def interval_x(self, text):
        try:
            interval_x = check_limits(text)
            self.onedimopt.update_interval_x(interval_x)
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            self.response.set_keyboard(Keyboards().for_optional_params())
            self.onedimopt.update_step('optional_params')
            return self.response
        except Exception as e:
            return self.error(e)

    def x0(self, text):
        try:
            x0 = check_float(text)
            self.onedimopt.update_x0(x0)
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
            self.onedimopt.update_step('optional_params')
            return self.response
        except Exception as e:
            return self.error(e)

    def input_optional_params(self, text, example):
        self.response.set_text(f'Введите {text}:\n\n'
                               f'Пример: {example}')
        return self.response

    def acc(self, text, task_type):
        try:
            acc = check_float(text)
            self.onedimopt.update_acc(acc)
            self.onedimopt.update_step('optional_params')
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            if task_type != 'bfgs':
                self.response.set_keyboard(Keyboards().for_optional_params())
            else:
                self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
            return self.response
        except Exception as e:
            return self.error(e)

    def max_iter(self, text, task_type):
        try:
            max_iter = check_int(text)
            self.onedimopt.update_max_iter(max_iter)
            self.onedimopt.update_step('optional_params')
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            if task_type != 'bfgs':
                self.response.set_keyboard(Keyboards().for_optional_params())
            else:
                self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
            return self.response
        except Exception as e:
            return self.error(e)

    def print_interim(self, text, task_type):
        if text == 'Да':
            self.onedimopt.update_print_interim(True)
        elif text == 'Нет':
            self.onedimopt.update_print_interim(False)
        else:
            return self.click_button()
        self.onedimopt.update_step('optional_params')
        self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
        if task_type != 'bfgs':
            self.response.set_keyboard(Keyboards().for_optional_params())
        else:
            self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
        return self.response

    def c1(self, text):
        try:
            c1 = check_float(text)
            self.onedimopt.update_c1(c1)
            self.onedimopt.update_step('optional_params')
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
            return self.response
        except Exception as e:
            return self.error(e)

    def c2(self, text):
        try:
            c2 = check_float(text)
            self.onedimopt.update_c2(c2)
            self.onedimopt.update_step('optional_params')
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
            return self.response
        except Exception as e:
            return self.error(e)

    def max_arg(self, text):
        try:
            max_arg = check_float(text)
            self.onedimopt.update_max_arg(max_arg)
            self.onedimopt.update_step('optional_params')
            self.response.set_text(Phrases.INPUT_OPTIONAL_PARAMS)
            self.response.set_keyboard(Keyboards().for_optional_params_bfgs())
            return self.response
        except Exception as e:
            return self.error(e)

    def input_print_interim(self):
        self.response.set_text(Phrases.INPUT_PRINT_INTERIM)
        self.response.set_keyboard(Keyboards().for_print_interim())
        return self.response

    def golden_ratio(self):
        params = {param: value for param, value in self.onedimopt.get_params().items() if value}
        params['func'] = prepare_func(params['func'])
        params['interval_x'] = prepare_limits(params['interval_x'])
        result = GoldenRatio(**params).solve()
        self.response.set_text(result)
        self.user.update_status('menu')
        self.response.set_keyboard(Keyboards().for_menu())
        return self.response

    def parabola(self):
        params = {param: value for param, value in self.onedimopt.get_params().items() if value}
        params['func'] = prepare_func(params['func'])
        params['interval_x'] = prepare_limits(params['interval_x'])
        result = Parabola(**params).solve()
        self.response.set_text(result)
        self.user.update_status('menu')
        self.response.set_keyboard(Keyboards().for_menu())
        return self.response

    def brandt(self):
        params = {param: value for param, value in self.onedimopt.get_params().items() if value}
        params['func'] = prepare_func(params['func'])
        params['interval_x'] = prepare_limits(params['interval_x'])
        result = Brandt(**params).solve()
        self.response.set_text(result)
        self.user.update_status('menu')
        self.response.set_keyboard(Keyboards().for_menu())
        return self.response

    def bfgs(self):
        params = {param: value for param, value in self.onedimopt.get_params().items() if value}
        params['func'] = prepare_func(params['func'])
        params['interval_x'] = prepare_limits(params['interval_x'])
        result = Parabola(**params).solve()
        self.response.set_text(result)
        self.user.update_status('menu')
        self.response.set_keyboard(Keyboards().for_menu())
        return self.response

    def error(self, error: Exception) -> Response:
        """
        Отправка сообщения об ошибке.

        Parameters
        ----------
        error : Exception
            Описание ошибки.

        Returns
        -------
        Response
            Сообщение для пользователя.
        """
        self.response.set_text(Phrases.ERROR.format(error))
        return self.response

    def click_button(self) -> Response:
        """
        Если пользователь ввёл непонятное сообщение.
        Статус: не переопределяется

        Returns
        -------
        Response
            Сообщение для пользователя.
        """

        self.response.set_text(Phrases.CLICK_BUTTON)
        return self.response




