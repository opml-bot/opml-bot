from vk_api.keyboard import VkKeyboard, VkKeyboardColor


class Keyboards:
    """
    Набор готовых клавиатур.
    """

    def __init__(self):
        self.keyboard = VkKeyboard(inline=True)

    def for_opt_alg_selection(self) -> VkKeyboard:
        """
        Выбор алгоритма для оптимизации.

        Returns
        -------
        VkKeyboard
            Объект созданной клавиатуры.
        """

        self.keyboard.add_button('Парабола', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Золотое сечение', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Брент', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('БФСГ', VkKeyboardColor.SECONDARY)
        return self.keyboard

    def for_optional_params(self) -> VkKeyboard:
        self.keyboard.add_button('Точность', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Максимум итераций', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Вывод промежуточных результатов', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Вычислить', VkKeyboardColor.POSITIVE)
        return self.keyboard

    def for_optional_params_bfgs(self) -> VkKeyboard:
        self.keyboard.add_button('Первое условие Вольфе', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Второе условие Вольфе', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Максимум аргумента', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Точность', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Максимум итераций', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Вывод промежуточных результатов', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Вычислить', VkKeyboardColor.POSITIVE)
        return self.keyboard

    def for_print_interim(self) -> VkKeyboard:
        self.keyboard.add_button('Да', VkKeyboardColor.POSITIVE)
        self.keyboard.add_button('Нет', VkKeyboardColor.NEGATIVE)
        return self.keyboard

    def for_menu(self) -> VkKeyboard:
        """
        Клавиатура для статуса menu.

        Returns
        -------
        VkKeyboard
            Объект созданной клавиатуры.
        """

        self.keyboard.add_button('Поиск экстремума', VkKeyboardColor.POSITIVE)
        self.keyboard.add_line()
        self.keyboard.add_button('Одномерная оптимизация', VkKeyboardColor.PRIMARY)
        self.keyboard.add_line()
        self.keyboard.add_button('Обо мне', VkKeyboardColor.SECONDARY)
        return self.keyboard