from typing import List

import cv2
import numpy as np

"""
Здесь может быть любой код, который необходим Вам для расчёта маски переднего плана на фото
这里可以是计算照片中前景蒙版所需的任何代码
"""


def get_foreground_mask(image_path: str) -> List[tuple]:
    """
    Метод для вычисления маски переднего плана на фото 一种计算照片中前景蒙版的方法
    :param image_path - путь до фото 照片路径
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    返回格式为[(x_1, y_1), (x_2, y_2), (x_3, y_3)]的数组，其中列出了与mask相关的所有点
    """

    # Read the image and convert it to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to set the foreground to white and background to black
    _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Perform morphological operations on the binary image to further process the foreground
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find the coordinates of non-zero pixels in the foreground mask
    pred_points = np.argwhere(binary_mask > 0)

    # Convert coordinates from [y, x] format to [(x, y)] format
    pred_points = [(point[1], point[0]) for point in pred_points]

    return pred_points
