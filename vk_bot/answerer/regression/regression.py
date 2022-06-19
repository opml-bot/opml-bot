from vk_bot.database import BotDatabase
from vk_bot.sql_queries import Select, Insert, Update


class Regression:
    def __init__(self, db: BotDatabase, user_id: int):
        self.db = db
        self.user_id = user_id

    def registration(self):
        """
        Регистрация пользователя в базе данных в таблице regression.
        """

        self.db.insert(Insert.REGRESSION, (self.user_id,))

    def get_step(self) -> str:
        """
        Извлечение шага, на котором находится пользователь при решении задачи.

        Returns
        -------
        str
            Шаг, на котором находится пользователь, для решения задачи.
        """

        if not self.db.select(Select.REGRESSION_STEP, (self.user_id,)):
            self.registration()
        return self.db.select(Select.REGRESSION_STEP, (self.user_id,))[0]

    def get_type(self) -> str:
        """
        Извлечение типа задачи, которая задается пользователем.

        Returns
        -------
        str
            Тип задачи, которая задается пользователем.
        """

        return self.db.select(Select.REGRESSION_TYPE, (self.user_id,))[0]

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

        self.db.update(Update.REGRESSION_STEP, (step, self.user_id))

    def update_type(self, task_type: str):
        """
        Изменение типа задачи в базе данных.

        Parameters
        ----------
        task_type : str
            Тип задачи, который будет записан в базу данных.
        """

        self.db.update(Update.REGRESSION_TYPE, (task_type, self.user_id))

    def update_degree(self, degree: int):
        """
        Изменение степени в базе данных.

        Parameters
        ----------
        func : int
            Степень, которая будет записана в базу данных.
        """

        self.db.update(Update.REGRESSION_DEGREE, (degree, self.user_id))

    def update_regularization(self, regularization: str):
        """
        Изменение  регуляризации в базе данных.

        Parameters
        ----------
        regularization : str
            Регуляризация, которая будет записан в базу данных.
        """

        self.db.update(Update.REGRESSION_REGULARIZATION, (regularization, self.user_id))

    def update_alpha(self, alpha: float):
        """
        Изменение альфа в базе данных.

        Parameters
        ----------
        alpha : float
            Альфа, которая будет записана в базу данных.
        """

        self.db.update(Update.REGRESSION_ALPHA, (alpha, self.user_id))
