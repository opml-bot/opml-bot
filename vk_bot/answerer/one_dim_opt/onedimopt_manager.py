from vk_api.vk_api import VkApiMethod

from vk_bot.answerer.one_dim_opt.message_handlers import Handlers
from vk_bot.answerer.one_dim_opt.onedimopt import OneDimOpt
from vk_bot.answerer.one_dim_opt.scripted_phrases import Examples
from vk_bot.answerer.response_init import Response
from vk_bot.database import BotDatabase
from vk_bot.user import User


class OneDimOptManager:
    """
    Менеджер управления решением задачи одномерной оптимизации.

    Parameters
    ----------
    vk_api_method : VkApiMethod
        Объект соединения с VK и набор методов API.
    db : BotDatabase
        Объект для взаимодействия с базой данных.
    user : User
        Объект для взаимодействия с данными пользователя.
    """

    def __init__(self, vk_api_method: VkApiMethod, db: BotDatabase, user: User):
        self.vk_api_method = vk_api_method
        self.db = db
        self.user = user
        self.onedimopt = OneDimOpt(db, user.user_id)
        self.step = self.onedimopt.get_step()
        self.type = self.onedimopt.get_type()
        self.handlers = Handlers(vk_api_method, db, user, self.onedimopt)

    def manage(self, text: str) -> Response:
        """
        Управление обработкой входящих сообщений для ввода исходных данных в задаче.

        Parameters
        ----------
        text : str
            Текст сообщения, отправленного пользователем.

        Returns
        -------
        Response
            Сообщение для пользователя.
        """

        if self.step == 'start':
            return self.handlers.opt_alg_selection()

        if self.step == 'opt_alg_selection':
            if text == 'Парабола':
                self.onedimopt.update_type('parabola')
            elif text == 'Золотое сечение':
                self.onedimopt.update_type('golden_ratio')
            elif text == 'Брент':
                self.onedimopt.update_type('brandt')
            elif text == 'БФСГ':
                self.onedimopt.update_type('bfsg')
            else:
                return self.handlers.click_button()
            return self.handlers.input_func()

        if self.step == 'input_func':
            return self.handlers.func(text, self.type)

        if self.step == 'input_interval_x':
            return self.handlers.interval_x(text)

        if self.step == 'input_x0':
            return self.handlers.x0(text)

        if self.step == 'input_c1':
            return self.handlers.c1(text)

        if self.step == 'input_c2':
            return self.handlers.c2(text)

        if self.step == 'input_max_arg':
            return self.handlers.max_arg(text)

        if self.step == 'input_acc':
            return self.handlers.acc(text, self.type)

        if self.step == 'input_max_iter':
            return self.handlers.max_iter(text, self.type)

        if self.step == 'input_print_interim':
            return self.handlers.print_interim(text, self.type)

        if self.step == 'optional_params':
            if text == 'Первое условие Вольфе':
                self.onedimopt.update_step('input_c1')
                return self.handlers.input_optional_params(text, Examples.C1)
            elif text == 'Второе условие Вольфе':
                self.onedimopt.update_step('input_c2')
                return self.handlers.input_optional_params(text, Examples.C2)
            elif text == 'Максимум аргумента':
                self.onedimopt.update_step('input_max_arg')
                return self.handlers.input_optional_params(text, Examples.MAX_ARG)
            elif text == 'Точность':
                self.onedimopt.update_step('input_acc')
                return self.handlers.input_optional_params(text, Examples.ACC)
            elif text == 'Максимум итераций':
                self.onedimopt.update_step('input_max_iter')
                return self.handlers.input_optional_params(text, Examples.MAX_ITER)
            elif text == 'Вывод промежуточных результатов':
                self.onedimopt.update_step('input_print_interim')
                return self.handlers.input_print_interim()
            elif text == 'Вычислить':
                if self.type == 'parabola':
                    return self.handlers.parabola()
                elif self.type == 'golden_ratio':
                    return self.handlers.golden_ratio()
                elif self.type == 'brandt':
                    return self.handlers.brandt()
                elif self.type == 'bfgs':
                    return self.handlers.bfgs()
            return self.handlers.click_button()
