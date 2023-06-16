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
    morphed  = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]


def mask_from_contours(ref_img, contours):
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
        mask = cv2.drawContours(mask, [contour], -1, (1,1,1), -1)
        
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



def get_brain_mask(image: np.array, bone_treshold: float=0.6, void_treshold: float=0.0,
                   bone_dilate_size: int=8, void_opening_size: int=10, void_dilate_size: int=10,
                   insides_opening_size: int=40, use_bone_convex_hull: bool=True, return_all: bool=False) -> np.array:
    """
    Выделение внутренностей мозга.
    
    Параметры
    ---------
    image : np.array
        Изображение.
    bone_treshold : float
        Порог детекции кости.
    void_treshold : float
        Порог детекции пустоты.
    bone_dilate_size : int
        Число пикселей, на которое расширяется маска кости.
    void_opening_size : int
        Минимальный размер изолированных участков, которые не будут удаляться из макси пустоты.
    void_dilate_size : int
        Число пикселей, на которое расширяется маска пустоты.
    """
    
    # Выделение кости по порогу.
    _, mask_bone = cv2.threshold(image, bone_treshold, 1.0, cv2.THRESH_BINARY)
    
    # Расширение на соседние пикселы.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bone_dilate_size, bone_dilate_size))
    mask_bone = cv2.dilate(mask_bone, kernel, iterations=1)
    
    # Выделение пустоты по порогу.
    _, mask_void = cv2.threshold(image, void_treshold, 1.0, cv2.THRESH_BINARY_INV)
    
    # Удаение шума.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_void = cv2.dilate(mask_void, kernel, iterations=1)
    
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
    mask = 1.0 - np.maximum(mask_bone, mask_void)
    
    if use_bone_convex_hull:
        # Выпуклая оболочка кости.
        mask_bone_convex_hull = convex_hull_from_mask(mask_bone)
        mask = np.minimum(mask, mask_bone_convex_hull)
    
    # Удаление мелких деталей.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (insides_opening_size, insides_opening_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    if return_all:
        return mask, mask_bone, mask_void
    else:
        return mask


def get_baseline_mask(image: np.array, min_treshold: float=0.17, max_treshold: float=0.4) -> np.array:
    """
    Базовое решение.
    
    Параметры
    ---------
    image : np.array
        Изображение.
    min_treshold : float
        Нижний порог обрезки кровоизлияния.
    max_treshold : float
        Верхний порог обрезки кровоизлияния.
    """
    
    # Выделение средних по интенсивности пикселей.
    _, mask_min = cv2.threshold(image, min_treshold, 1.0, cv2.THRESH_BINARY)
    _, mask_max = cv2.threshold(image, max_treshold, 1.0, cv2.THRESH_BINARY_INV)
    mask = np.minimum(mask_min, mask_max)
    
    # Выделение кости.
    bone_treshold = 0.4
    kernel_size = 10
    
    _, mask_bone = cv2.threshold(image, bone_treshold, 1.0, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_bone = cv2.dilate(mask_bone, kernel, iterations=1)
    
    # Выделение внешнего пространства.
    void_treshold = 0.02
    erode_kernel_size = 10
    dilate_kernel_size = 50
    
    _, mask_void = cv2.threshold(image, void_treshold, 1.0, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    mask_void = cv2.erode(mask_void, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    mask_void = cv2.dilate(mask_void, kernel, iterations=1)
    
    # Обрезка всего, что не касается мозга.
    mask_cut = np.maximum(mask_bone, mask_void)
    mask[mask_cut > 0.0] = 0.0
    
    # Удаление мелких деталей.
    erode_kernel_size = 6
    dilate_kernel_size = 7
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel_size, erode_kernel_size))
    mask = cv2.erode(mask, kernel, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def plot_with_mask(image: np.array, mask: np.array, size: float=8) -> None:
    """
    Вывести подряд избражение и маску.
    
    Параметры
    ---------
    image : np.array
        Изображение.
    mask : np.array
        Макска.
    size : float, optional
        Размер изображения.
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(2*size, size))
    
    axes[0].imshow(image)
    axes[1].imshow(np.maximum(image, mask))
    
    plt.show()