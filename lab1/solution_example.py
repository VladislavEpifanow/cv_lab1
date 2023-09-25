from typing import List

import cv2
import numpy as np


def euclid_distance(point_1, point_2):
    return np.linalg.norm(point_1 - point_2)


def get_foreground_mask(image_path: str) -> List[tuple]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img_color = cv2.imread(image_path)

    gray_img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # Thresholding image
    thr_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # Find all object pixels found
    kernel = np.ones((5, 5), np.uint8)
    thr_img = cv2.morphologyEx(thr_img, cv2.MORPH_CLOSE, kernel)
    thr_img = cv2.morphologyEx(thr_img, cv2.MORPH_OPEN, kernel)

    # Calculate which of two pixels group is nearest to center with euclid distance
    img_x_center = img_color.shape[0] / 2
    img_y_center = img_color.shape[1] / 2
    img_center = np.array([img_x_center, img_y_center])
    first_points = np.argwhere(thr_img > 0)
    second_points = np.argwhere(thr_img < 1)
    if len(second_points) == 0:
        return first_points
    elif len(first_points) == 0:
        return second_points

    else:
        first_points_euclid_dist = 0
        second_points_euclid_dist = 0
        for x, y in first_points:
            temp_arr = np.array([x, y])
            first_points_euclid_dist += euclid_distance(temp_arr, img_center)
        first_points_euclid_dist /= first_points.shape[0] + 1

        for x, y in second_points:
            temp_arr = np.array([x, y])
            second_points_euclid_dist += euclid_distance(temp_arr, img_center)
        second_points_euclid_dist /= second_points.shape[0] + 1

        if first_points_euclid_dist > second_points_euclid_dist:
            front_obj = second_points
        else:
            front_obj = first_points

    return front_obj