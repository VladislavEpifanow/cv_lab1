from typing import List

import cv2
import numpy as np

"""
Здесь может быть любой код, который необходим Вам для расчёта маски переднего плана на фото
"""


def get_foreground_mask(image_path: str) -> List[tuple]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # your code next
    ###

    # example submission
    pred_points = np.argwhere(img)
    return pred_points