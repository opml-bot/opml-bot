from typing import NamedTuple


class Create(NamedTuple):
    """
    Запросы на создание таблиц в базе данных.
    """
    USERS = ("CREATE TABLE users (\n"
             "               user_id INTEGER PRIMARY KEY, \n"
             "               first_name TEXT NOT NULL, \n"
             "               last_name TEXT NOT NULL, \n"
             "               status TEXT DEFAULT 'start')")

    EXTREMES = ("CREATE TABLE extremes (\n"
                "                  user_id INTEGER PRIMARY KEY, \n"
                "                  step TEXT DEFAULT 'start',\n"
                "                  type TEXT,\n"
                "                  vars TEXT, \n"
                "                  func TEXT, \n"
                "                  interval_x TEXT, \n"
                "                  interval_y TEXT, \n"
                "                  g_func TEXT, \n"
                "                  restr INTEGER, \n"
                "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")

    ONEDIMOPT = ("CREATE TABLE onedimopt (\n"
                 "                  user_id INTEGER PRIMARY KEY, \n"
                 "                  step TEXT DEFAULT 'start', \n"
                 "                  type TEXT, \n"
                 "                  func TEXT, \n"
                 "                  interval_x TEXT, \n"
                 "                  acc REAL, \n"
                 "                  max_iter INTEGER, \n"
                 "                  print_interim INTEGER, \n"
                 "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")

    GRAD = ("CREATE TABLE extremes (\n"
            "                  user_id INTEGER PRIMARY KEY, \n"
            "                  step TEXT DEFAULT 'start',\n"
            "                  type TEXT,\n"
            "                  func TEXT, \n"
            "                  interval_x TEXT, \n"
            "                  acc REAL, \n"
            "                  max_iteration INTEGER, \n"
            "                  gradient TEXT, \n"
            "                  started_point TEXT, \n"
            "                  alpha REAL, \n"
            "                  print_midterm TEXT, \n"
            "                  delta REAL, \n"
            "                  jac TEXT, \n"
            "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")

    REGRESSION = ("CREATE TABLE extremes (\n"
                  "                  user_id INTEGER PRIMARY KEY, \n"
                  "                  step TEXT DEFAULT 'start',\n"
                  "                  type TEXT,\n"
                  "                  degree INTEGER, \n"
                  "                  regularization  TEXT, \n"
                  "                  alpha REAL, \n"
                  "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")

    INNERPOINT = ("CREATE TABLE extremes (\n"
                  "                  user_id INTEGER PRIMARY KEY, \n"
                  "                  step TEXT DEFAULT 'start',\n"
                  "                  type TEXT,\n"
                  "                  function TEXT, \n"
                  "                  restrictions TEXT, \n"
                  "                  x0 TEXT, \n"
                  "                  eps REAL, \n"
                  "                  restr_uneq TEXT, \n"
                  "                  k REAL, \n"
                  "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")

    CLASSIFICATION = ("CREATE TABLE extremes (\n"
                      "                  user_id INTEGER PRIMARY KEY, \n"
                      "                  step TEXT DEFAULT 'start',\n"
                      "                  type TEXT,\n"
                      "                  max_iter INTEGER, \n"
                      "                  type_cls TEXT, \n"
                      "                  degree INTEGER, \n"
                      "                  draw_flag TEXT, \n"
                      "                  delta_w INTEGER, \n"
                      "                  alpha REAL, \n"
                      "                  regularization TEXT, \n"
                      "                  mu REAL, \n"
                      "                  c REAL, \n"
                      "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")

    INTEGERLP = ("CREATE TABLE extremes (\n"
                 "                  user_id INTEGER PRIMARY KEY, \n"
                 "                  step TEXT DEFAULT 'start',\n"
                 "                  type TEXT,\n"
                 "                  num_vars INT, \n"
                 "                  constraints  TEXT, \n"
                 "                  objective_function TXT, \n"
                 "                  FOREIGN KEY(user_id) REFERENCES users(user_id))")


class Insert(NamedTuple):
    """
    Запросы на добавление новых строк в таблицу базы данных.
    """

    USERS = "INSERT INTO users(user_id, first_name, last_name) VALUES (?, ?, ?)"
    EXTREMES = "INSERT INTO extremes(user_id) VALUES (?)"
    ONEDIMOPT = "INSERT INTO onedimopt(user_id) VALUES (?)"
    GRAD = "INSERT INTO grad(user_id) VALUES (?)"
    REGRESSION = "INSERT INTO regression(user_id) VALUES (?)"
    INNERPOINT = "INSERT INTO innerpoint(user_id) VALUES (?)"
    CLASSIFICATION = "INSERT INTO classification(user_id) VALUES (?)"
    INTEGERLP = "INSERT INTO integerlp(user_id) VALUES (?)"


class Select(NamedTuple):
    """
    Запросы на извлечение полей из таблицы базы данных.
    """

    USERS_USER_ID = "SELECT user_id FROM users WHERE user_id = ?"
    USERS_STATUS = "SELECT status FROM users WHERE user_id = ?"

    EXTREMES_STEP = "SELECT step FROM extremes WHERE user_id = ?"
    EXTREMES_RESTR = "SELECT restr FROM extremes WHERE user_id = ?"
    EXTREMES_VARS = "SELECT vars FROM extremes WHERE user_id = ?"
    EXTREMES_TYPE = "SELECT type FROM extremes WHERE user_id = ?"
    EXTREMES_ALL = "SELECT vars, func, g_func, restr, interval_x, interval_y FROM extremes WHERE user_id = ?"
    EXTREMES_WITH_INT = "SELECT vars, func, interval_x, interval_y FROM extremes WHERE user_id = ?"
    EXTREMES_WITHOUT_INT = "SELECT vars, func FROM extremes WHERE user_id = ?"
    EXTREMES_RESTR_WITH_INT = "SELECT vars, func, g_func, interval_x, interval_y FROM extremes WHERE user_id = ?"
    EXTREMES_RESTR_WITHOUT_INT = "SELECT vars, func, g_func FROM extremes WHERE user_id = ? "

    ONEDIMOPT_STEP = "SELECT step FROM onedimopt WHERE user_id = ?"
    ONEDIMOPT_TYPE = "SELECT type FROM onedimopt WHERE user_id = ?"
    ONEDIMOPT_ALL = "SELECT * FROM onedimopt WHERE user_id = ?"

    GRAD_STEP = "SELECT step FROM grad WHERE user_id = ?"
    GRAD_TYPE = "SELECT type FROM grad WHERE user_id = ?"
    GRAD_ALL = "SELECT * FROM grad WHERE user_id = ?"

    REGRESSION_STEP = "SELECT step FROM regression WHERE user_id = ?"
    REGRESSION_TYPE = "SELECT type FROM regression WHERE user_id = ?"
    REGRESSION_ALL = "SELECT * FROM regression WHERE user_id = ?"

    INNERPOINT_STEP = "SELECT step FROM innerpoint WHERE user_id = ?"
    INNERPOINT_TYPE = "SELECT type FROM innerpoint WHERE user_id = ?"
    INNERPOINT_ALL = "SELECT * FROM innerpoint WHERE user_id = ?"

    CLASSIFICATION_STEP = "SELECT step FROM classification WHERE user_id = ?"
    CLASSIFICATION_TYPE = "SELECT type FROM classification WHERE user_id = ?"
    CLASSIFICATION_ALL = "SELECT * FROM classification WHERE user_id = ?"

    INTEGERLP_STEP = "SELECT step FROM integerlp WHERE user_id = ?"
    INTEGERLP_TYPE = "SELECT type FROM integerlp WHERE user_id = ?"
    INTEGERLP_ALL = "SELECT * FROM integerlp WHERE user_id = ?"


class Update(NamedTuple):
    """
    Запросы на обновление полей в таблицах базы данных.
    """

    USERS_STATUS = "UPDATE users SET status = ? WHERE user_id = ?"

    EXTREMES_STEP = "UPDATE extremes SET step = ? WHERE user_id = ?"
    EXTREMES_TYPE = "UPDATE extremes SET type = ? WHERE user_id = ?"
    EXTREMES_VARS = "UPDATE extremes SET vars = ? WHERE user_id = ?"
    EXTREMES_FUNC = "UPDATE extremes SET func = ? WHERE user_id = ?"
    EXTREMES_G_FUNC = "UPDATE extremes SET g_func = ? WHERE user_id = ?"
    EXTREMES_RESTR = "UPDATE extremes SET restr = ? WHERE user_id = ?"
    EXTREMES_INTERVAL_X = "UPDATE extremes SET interval_x = ? WHERE user_id = ?"
    EXTREMES_INTERVAL_Y = "UPDATE extremes SET interval_y = ? WHERE user_id = ?"

    ONEDIMOPT_STEP = "UPDATE onedimopt SET step = ? WHERE user_id = ?"
    ONEDIMOPT_TYPE = "UPDATE onedimopt SET type = ? WHERE user_id = ?"
    ONEDIMOPT_FUNC = "UPDATE onedimopt SET func = ? WHERE user_id = ?"
    ONEDIMOPT_INTERVAL_X = "UPDATE onedimopt SET interval_x = ? WHERE user_id = ?"
    ONEDIMOPT_ACC = "UPDATE onedimopt SET acc = ? WHERE user_id = ?"
    ONEDIMOPT_MAX_ITER = "UPDATE onedimopt SET max_iter = ? WHERE user_id = ?"
    ONEDIMOPT_PRINT_INTERIM = "UPDATE onedimopt SET print_interim = ? WHERE user_id = ?"

    GRAD_STEP = "UPDATE grad SET step = ? WHERE user_id = ?"
    GRAD_TYPE = "UPDATE grad SET type = ? WHERE user_id = ?"
    GRAD_FUNC = "UPDATE grad SET func = ? WHERE user_id = ?"
    GRAD_INTERVAL_X = "UPDATE grad SET interval_x = ? WHERE user_id = ?"
    GRAD_ACC = "UPDATE grad SET acc = ? WHERE user_id = ?"
    GRAD_MAX_ITERATION = "UPDATE grad SET max_iteration = ? WHERE user_id = ?"
    GRAD_GRADIENT = "UPDATE grad SET gradient = ? WHERE user_id = ?"
    GRAD_STARTED_POINT = "UPDATE grad SET started_point = ? WHERE user_id = ?"
    GRAD_ALPHA = "UPDATE grad SET alpha = ? WHERE user_id = ?"
    GRAD_PRINT_MIDTERM = "UPDATE grad SET print_midterm = ? WHERE user_id = ?"
    GRAD_DELTA = "UPDATE grad SET delta = ? WHERE user_id = ?"
    GRAD_JAC = "UPDATE grad SET jac = ? WHERE user_id = ?"

    REGRESSION_STEP = "UPDATE regression SET step = ? WHERE user_id = ?"
    REGRESSION_TYPE = "UPDATE regression SET type = ? WHERE user_id = ?"
    REGRESSION_DEGREE = "UPDATE regression SET degree = ? WHERE user_id = ?"
    REGRESSION_REGULARIZATION = "UPDATE regression SET regularization  = ? WHERE user_id = ?"
    REGRESSION_ALPHA = "UPDATE regression SET alpha = ? WHERE user_id = ?"

    INNERPOINT_STEP = "UPDATE innerpoint SET step = ? WHERE user_id = ?"
    INNERPOINT_TYPE = "UPDATE innerpoint SET type = ? WHERE user_id = ?"
    INNERPOINT_FUNCTION = "UPDATE innerpoint SET function = ? WHERE user_id = ?"
    INNERPOINT_RESTRICTIONS = "UPDATE innerpoint SET restrictions  = ? WHERE user_id = ?"
    INNERPOINT_X0 = "UPDATE innerpoint SET x0 = ? WHERE user_id = ?"
    INNERPOINT_EPS = "UPDATE innerpoint SET eps = ? WHERE user_id = ?"
    INNERPOINT_RESTR_UNEQ = "UPDATE innerpoint SET restr_uneq = ? WHERE user_id = ?"
    INNERPOINT_K = "UPDATE innerpoint SET k = ? WHERE user_id = ?"

    CLASSIFICATION_STEP = "UPDATE classification SET step = ? WHERE user_id = ?"
    CLASSIFICATION_TYPE = "UPDATE classification SET type = ? WHERE user_id = ?"
    CLASSIFICATION_MAX_ITER = "UPDATE classification SET max_iter = ? WHERE user_id = ?"
    CLASSIFICATION_TYPE_CLS = "UPDATE classification SET type_cls  = ? WHERE user_id = ?"
    CLASSIFICATION_DEGREE = "UPDATE classification SET degree = ? WHERE user_id = ?"
    CLASSIFICATION_DRAW_FLAG = "UPDATE classification SET draw_flag = ? WHERE user_id = ?"
    CLASSIFICATION_DELTA_W = "UPDATE classification SET delta_w = ? WHERE user_id = ?"
    CLASSIFICATION_ALPHA = "UPDATE classification SET alpha = ? WHERE user_id = ?"
    CLASSIFICATION_REGULARIZATION = "UPDATE classification SET regularization = ? WHERE user_id = ?"
    CLASSIFICATION_MU = "UPDATE classification SET mu = ? WHERE user_id = ?"
    CLASSIFICATION_C = "UPDATE classification SET c = ? WHERE user_id = ?"

    INTEGERLP_STEP = "UPDATE integerlp SET step = ? WHERE user_id = ?"
    INTEGERLP_TYPE = "UPDATE integerlp SET type = ? WHERE user_id = ?"
    INTEGERLP_NUM_VARS = "UPDATE integerlp SET num_vars = ? WHERE user_id = ?"
    INTEGERLP_CONSTRAINTS = "UPDATE integerlp SET constraints  = ? WHERE user_id = ?"
    INTEGERLP_OBJECTIVE_FUNCTION = "UPDATE integerlp SET objective_function = ? WHERE user_id = ?"