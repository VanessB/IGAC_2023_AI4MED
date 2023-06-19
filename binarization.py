import numpy as np


def Otsu_criterion(array: np.array, treshold: float) -> float:
    """
    Критерий Otsu качества разбиения массива чисел на две части.
    
    Параметры
    ---------
    array : np.array
        Отсортированный массив.
    treshold : float
        Порог разбиения.
    """
    
    index = np.searchsorted(array, treshold)
    if 0 < index < len(array):
        return np.var(array[:index]) + np.var(array[index:])
    else:
        return np.var(array)


def find_best_treshold(array: np.array, criterion: callable) -> float:
    """
    Поиск лучшего порога бинаризации по заданному критерию.
    
    Параметры
    ---------
    array : np.array
        Массив, который требуется бинаризировать.
    criterion : callable
        Критерий подбора оптимального порога (выдаёт меньшие значения для лучшего порога).
    """
    
    array = np.sort(array)
    unique = np.append(np.unique(array), array[-1] + 1)
    
    criterion_values = np.array([criterion(array, treshold) for treshold in unique])
    #print(unique)
    #print(criterion_values)
    return unique[np.argmin(criterion_values)]