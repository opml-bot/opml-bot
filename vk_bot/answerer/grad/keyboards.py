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

        self.keyboard.add_button('Алгоритм "Градиентый спуск с постоянным шагом"', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Алгоритм "Градиентный спуск с дроблением шага"', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Алгоритм "Наискорейший градиентный спуск "', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Алгоритм "Ньютона-сопряженного градиента"', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Алгоритм "Комбинированный метод Брента"', VkKeyboardColor.PRIMARY)

        # TODO: добавить кнопки для остальных алгоритмов
        return self.keyboard