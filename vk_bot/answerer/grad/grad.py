from vk_bot.database import BotDatabase
from vk_bot.sql_queries import Select, Insert, Update


class Grad:
    def __init__(self, db: BotDatabase, user_id: int):
        self.db = db
        self.user_id = user_id

    def registration(self):
        """
        Регистрация пользователя в базе данных в таблице grad.
        """

        self.db.insert(Insert.GRAD, (self.user_id,))

    def get_step(self) -> str:
        """
        Извлечение шага, на котором находится пользователь при решении задачи.

        Returns
        -------
        str
            Шаг, на котором находится пользователь, для решения задачи.
        """

        if not self.db.select(Select.GRAD_STEP, (self.user_id,)):
            self.registration()
        return self.db.select(Select.GRAD_STEP, (self.user_id,))[0]

    def get_type(self) -> str:
        """
        Извлечение типа задачи, которая задается пользователем.

        Returns
        -------
        str
            Тип задачи, которая задается пользователем.
        """

        return self.db.select(Select.GRAD_TYPE, (self.user_id,))[0]

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

        self.db.update(Update.GRAD_STEP, (step, self.user_id))

    def update_type(self, task_type: str):
        """
        Изменение типа задачи в базе данных.

        Parameters
        ----------
        task_type : str
            Тип задачи, который будет записан в базу данных.
        """

        self.db.update(Update.GRAD_TYPE, (task_type, self.user_id))

    def update_func(self, func: str):
        """
        Изменение функции в базе данных.

        Parameters
        ----------
        func : str
            Функция, которая будет записана в базу данных.
        """

        self.db.update(Update.GRAD_FUNC, (func, self.user_id))

    def update_interval_x(self, interval_x: str):
        """
        Изменение интервала X для функции в базе данных.

        Parameters
        ----------
        interval_x : str
            Интервал X для функции, который будет записан в базу данных.
        """

        self.db.update(Update.GRAD_INTERVAL_X, (interval_x, self.user_id))

    def update_acc(self, acc: float):
        """
        Изменение целевой точности для алгоритма в базе данных.

        Parameters
        ----------
        acc : float
            Целевая точность для алгоритма, которая будет записана в базу данных.
        """

        self.db.update(Update.GRAD_ACC, (acc, self.user_id))

    def update_max_iteration(self, max_iteration: int):
        """
        Изменение максимального количества итераций алгоритма в базе данных.

        Parameters
        ----------
        max_iteration : int
            Максимальное количество итераций алгоритма, которое будет записано в базу данных.
        """

        self.db.update(Update.GRAD_MAX_ITERATION, (max_iteration, self.user_id))

    def update_gradient(self, gradient: str):
        """
        Изменение градиента в базе данных.

        Parameters
        ----------
        gradient : str
             градиент, который будет записан в базу данных.
        """

        self.db.update(Update.GRAD_GRADIENT, (gradient, self.user_id))

    def update_started_point(self, started_point: str):
        """
        Изменение начальной точки в базе данных.

        Parameters
        ----------
        started_point : str
             начальная точка, которая будет записан в базу данных.
        """

        self.db.update(Update.GRAD_STARTED_POINT, (started_point, self.user_id)

    def update_alpha(self, alpha: float):
        """
        Изменение альфа в базе данных.

        Parameters
        ----------
        alpha : float
             альфа, которая будет записан в базу данных.
        """

        self.db.update(Update.GRAD_ALPHA, (alpha, self.user_id))

    def update_print_midterm(self, print_midterm: str):
        """
        Изменение вывода промежуточных результатов в базе данных.

        Parameters
        ----------
        print_midterm : float
             промежуточный результат, который будет записан в базу данных.
        """

        self.db.update(Update.GRAD_PRINT_MIDTERM, (print_midterm, self.user_id))

    def update_delta(self, delta: float):
        """
        Изменение дельта в базе данных.

        Parameters
        ----------
        delta : float
             дельта, которая будет записан в базу данных.
        """

        self.db.update(Update.GRAD_DELTA, (delta, self.user_id))

    def update_jac(self, jac: str):
        """
        Изменение якобиана в базе данных.

        Parameters
        ----------
        jac : str
             якобиан, который будет записан в базу данных.
        """

        self.db.update(Update.GRAD_JAC, (jac, self.user_id))