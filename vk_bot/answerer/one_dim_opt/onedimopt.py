from vk_bot.database import BotDatabase
from vk_bot.sql_queries import Select, Insert, Update


class OneDimOpt:
    def __init__(self, db: BotDatabase, user_id: int):
        self.db = db
        self.user_id = user_id

    def registration(self):
        """
        Регистрация пользователя в базе данных в таблице onedimopt.
        """

        self.db.insert(Insert.ONEDIMOPT, (self.user_id,))

    def get_step(self) -> str:
        """
        Извлечение шага, на котором находится пользователь при решении задачи.

        Returns
        -------
        str
            Шаг, на котором находится пользователь, для решения задачи.
        """

        if not self.db.select(Select.ONEDIMOPT_STEP, (self.user_id,)):
            self.registration()
        return self.db.select(Select.ONEDIMOPT_STEP, (self.user_id,))[0]

    def get_type(self) -> str:
        """
        Извлечение типа задачи, которая задается пользователем.

        Returns
        -------
        str
            Тип задачи, которая задается пользователем.
        """

        return self.db.select(Select.ONEDIMOPT_TYPE, (self.user_id,))[0]

    def get_params(self) -> dict:
        """
        Извлечение всех внесенных данных для передачи их в реализацию алгоритма.

        Returns
        -------
        dict
            Все данные, необходимые для решения задачи.
        """

        all_params = self.db.select(Select.ONEDIMOPT_ALL, (self.user_id,))
        params = {
            'func': all_params[3],
            'interval_x': all_params[4],
            'x0': all_params[5],
            'c1': all_params[6],
            'c2': all_params[7],
            'max_arg': all_params[8],
            'acc': all_params[9],
            'max_iter': all_params[10],
            'print_interim': all_params[11]
        }
        return params

    def update_step(self, step: str):
        """
        Изменение шага, на котором находится пользователь, в базе данных.

        Parameters
        ----------
        step : str
            Новый шаг, который будет записан в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_STEP, (step, self.user_id))

    def update_type(self, task_type: str):
        """
        Изменение типа задачи в базе данных.

        Parameters
        ----------
        task_type : str
            Тип задачи, который будет записан в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_TYPE, (task_type, self.user_id))

    def update_func(self, func: str):
        """
        Изменение функции в базе данных.

        Parameters
        ----------
        func : str
            Функция, которая будет записана в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_FUNC, (func, self.user_id))

    def update_interval_x(self, interval_x: str):
        """
        Изменение интервала X для функции в базе данных.

        Parameters
        ----------
        interval_x : str
            Интервал X для функции, который будет записан в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_INTERVAL_X, (interval_x, self.user_id))

    def update_x0(self, x0: float):
        self.db.update(Update.ONEDIMOPT_X0, (x0, self.user_id))

    def update_c1(self, c1: float):
        self.db.update(Update.ONEDIMOPT_C1, (c1, self.user_id))

    def update_c2(self, c2: float):
        self.db.update(Update.ONEDIMOPT_C2, (c2, self.user_id))

    def update_max_arg(self, max_arg: float):
        self.db.update(Update.ONEDIMOPT_MAX_ARG, (max_arg, self.user_id))

    def update_acc(self, acc: float):
        """
        Изменение целевой точности для алгоритма в базе данных.

        Parameters
        ----------
        acc : float
            Целевая точность для алгоритма, которая будет записана в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_ACC, (acc, self.user_id))

    def update_max_iter(self, max_iter: int):
        """
        Изменение максимального количества итераций алгоритма в базе данных.

        Parameters
        ----------
        max_iter : int
            Максимальное количество итераций алгоритма, которое будет записано в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_MAX_ITER, (max_iter, self.user_id))

    def update_print_interim(self, print_interim: int):
        """
        Изменение флага вывода промежуточных результатов в базе данных.

        Parameters
        ----------
        print_interim : int
            Флаг вывода промежуточных результатов, который будет записан в базу данных.
        """

        self.db.update(Update.ONEDIMOPT_PRINT_INTERIM, (print_interim, self.user_id))
