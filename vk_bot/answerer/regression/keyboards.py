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

        self.keyboard.add_button('Алгоритм "Линейная регрессия"', VkKeyboardColor.PRIMARY)
        self.keyboard.add_button('Алгоритм "Полиномиальная регрессия"', VkKeyboardColor.SECONDARY)
        self.keyboard.add_button('Алгоритм "Экспоненциальная регрессия"', VkKeyboardColor.PRIMARY)

        # TODO: добавить кнопки для остальных алгоритмов
        return self.keyboard
