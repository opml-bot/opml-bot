from vk_bot.database import BotDatabase
from vk_bot.sql_queries import Select, Insert, Update


class InnerPoint:
    def __init__(self, db: BotDatabase, user_id: int):
        self.db = db
        self.user_id = user_id

    def registration(self):
        """
        Регистрация пользователя в базе данных в таблице innerpoint.
        """

        self.db.insert(Insert.INNERPOINT, (self.user_id,))

    def get_step(self) -> str:
        """
        Извлечение шага, на котором находится пользователь при решении задачи.

        Returns
        -------
        str
            Шаг, на котором находится пользователь, для решения задачи.
        """

        if not self.db.select(Select.INNERPOINT_STEP, (self.user_id,)):
            self.registration()
        return self.db.select(Select.INNERPOINT_STEP, (self.user_id,))[0]

    def get_type(self) -> str:
        """
        Извлечение типа задачи, которая задается пользователем.

        Returns
        -------
        str
            Тип задачи, которая задается пользователем.
        """

        return self.db.select(Select.INNERPOINT_TYPE, (self.user_id,))[0]

    def get_params(self) -> str:
        """
        Извлечение всех внесенных данных для передачи их в реализацию алгоритма.

        Returns
        -------
        str
            Все данные, необходимые для решения задачи.
        """

        # TODO: написать реализацию метода
        pass

    def update_step(self, step: str):
        """
        Изменение шага, на котором находится пользователь, в базе данных.

        Parameters
        ----------
        step : str
            Новый шаг, который будет записан в базу данных.
        """

        self.db.update(Update.INNERPOINT_STEP, (step, self.user_id))

    def update_type(self, task_type: str):
        """
        Изменение типа задачи в базе данных.

        Parameters
        ----------
        task_type : str
            Тип задачи, который будет записан в базу данных.
        """

        self.db.update(Update.INNERPOINT_TYPE, (task_type, self.user_id))

    def update_function(self, function: str):
        """
        Изменение функции в базе данных.

        Parameters
        ----------
        function : str
            Функция, которая будет записана в базу данных.
        """

        self.db.update(Update.INNERPOINT_FUNCTION, (function, self.user_id))

    def update_restrictions(self, restrictions: str):
        """
        Изменение ограничений в базе данных.

        Parameters
        ----------
        restrictions : str
            Ограничения , которые будут записаны в базу данных.
        """

        self.db.update(Update.INNERPOINT_RESTRICTIONS, (restrictions, self.user_id))

    def update_x0(self, x0: str):
        """
        Изменение начальной точки для алгоритма в базе данных.

        Parameters
        ----------
        x0 : str
            Начальная точка для алгоритма, которая будет записана в базу данных.
        """

        self.db.update(Update.INNERPOINT_X0, (x0, self.user_id))

    def update_eps(self, eps: float):
        """
        Изменение точности алгоритма в базе данных.

        Parameters
        ----------
        eps : float
            Точность алгоритма, которае будет записана в базу данных.
        """

        self.db.update(Update.INNERPOINT_EPS, (eps, self.user_id))

    def update_restr_uneq(self, restr_uneq: str):
        """
        Изменение списка вызываемых функций в базе данных.

        Parameters
        ----------
        restr_uneq : str
            Список вызываемых функций, который будет записан в базу данных.
        """

        self.db.update(Update.INNERPOINT_RESTR_UNEQ, (restr_uneq, self.user_id))

    def update_k(self, k: float):
        """
        Изменение параметра k для алгоритма в базе данных.

        Parameters
        ----------
        k : float
            Параметр k для алгоритма, которsq будет записан в базу данных.
        """

        self.db.update(Update.INNERPOINT_K, (k, self.user_id))