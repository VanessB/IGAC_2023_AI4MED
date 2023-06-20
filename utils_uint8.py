import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def find_contours(mask: np.array) -> np.array:
    """
    Получение контуров по маске.
    
    Параметры
    ---------
    mask : np.array
        Маска.
    """
    
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]


def mask_from_contours(ref_img: np.array, contours: list) -> np.array:
    """
    Построение маски по контурам.
    
    Параметры
    ---------
    ref_img : np.array
        Референсное изображение (нужно для определение размеров).
    contours : list
        Контуры.
    """
    
    mask = np.zeros(ref_img.shape, np.uint8)
    for contour in contours:
        mask = cv2.drawContours(mask, [contour], -1, (255,255,255), -1)
        
    return mask


def convex_hull_from_mask(mask: np.array) -> np.array:
    """
    Получение выпуклой оболочки маски.
    
    Параметры
    ---------
    mask : np.array
        Маска.
    """
    
    contours = find_contours(mask)
    hull = []
    for countour in contours:
        hull.append(cv2.convexHull(countour, False))
    
    return mask_from_contours(mask, hull)



def get_brain_mask(image: np.array, bone_treshold: int=153, void_treshold: int=0,
                   bone_dilate_size: int=8, void_opening_size: int=10, void_dilate_size: int=10,
                   insides_opening_size: int=40, use_bone_convex_hull: bool=True, return_all: bool=False) -> np.array:
    """
    Выделение внутренностей мозга.
    
    Параметры
    ---------
    image : np.array
        Изображение.
    bone_treshold : int
        Порог детекции кости.
    void_treshold : int
        Порог детекции пустоты.
    bone_dilate_size : int
        Число пикселей, на которое расширяется маска кости.
    void_opening_size : int
        Минимальный размер изолированных участков, которые не будут удаляться из макси пустоты.
    void_dilate_size : int
        Число пикселей, на которое расширяется маска пустоты.
    use_bone_convex_hull : bool
        Пересечь полученную маску с выпуклой оболочкой кости.
    return_all : bool
        Вернуть все промежуточные маски.
    """
    
    # Выделение кости по порогу.
    _, mask_bone = cv2.threshold(image, bone_treshold, 255, cv2.THRESH_BINARY)
    
    # Расширение на соседние пикселы.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bone_dilate_size, bone_dilate_size))
    mask_bone = cv2.dilate(mask_bone, kernel)
    
    # Выделение пустоты по порогу.
    _, mask_void = cv2.threshold(image, void_treshold, 255, cv2.THRESH_BINARY_INV)
    
    # Удаение шума.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_void = cv2.dilate(mask_void, kernel)
    
    # Удаление кусков размера меньше void_opening_size.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (void_opening_size, void_opening_size))
    mask_void = cv2.morphologyEx(mask_void, cv2.MORPH_OPEN, kernel)
    
    # Точно внешнее пространство (не полости внутри челюсти/глаз/...).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    mask_big_void = cv2.morphologyEx(mask_void, cv2.MORPH_OPEN, kernel)
    
    # Расширение на соседние пикселы.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (void_dilate_size, void_dilate_size))
    mask_void = cv2.dilate(mask_void, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask_big_void = cv2.dilate(mask_big_void, kernel, iterations=1)
    
    mask_void = np.maximum(mask_void, mask_big_void)
    
    # Совмещение масок.
    mask = 255 - np.maximum(mask_bone, mask_void)
    
    if use_bone_convex_hull:
        # Выпуклая оболочка кости.
        mask_bone_convex_hull = convex_hull_from_mask(mask_bone)
        mask = np.minimum(mask, mask_bone_convex_hull)
    
    # Удаление мелких деталей.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (insides_opening_size, insides_opening_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if return_all:
        return mask, mask_bone, mask_void
    else:
        return mask