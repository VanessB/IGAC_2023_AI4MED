# Innopolis Global AI Challenge 2023: AI4MED
Решение команды "Салават Юлаев".

## Структура репозитория

- [`utils.py`](./utils.py) - вспомогательные функции, а также базовое решение.
- [`utils_uint8.py`](./utils_uint8.py) - вспомогательные функции для масок в формате uint8.
- [`Baseline.ipynb`](./Baseline.ipynb) - демонстрация базового решения.
- [`CheckBrainMask.ipynb`](./CheckBrainMask.ipynb) - визуальная проверка качества выделения мозга.
- [`ImprovedBaseline.ipynb`](./ImprovedBaseline.ipynb) - улучшенное базовое решение.
- [`ConnectedComponents.ipynb`](./ConnectedComponents.ipynb) - решение, основанное на анализе компонент связности **(используется для генерации посылок)**.
- [`ConnectedComponentsLite.ipynb`](./ConnectedComponentsLite.ipynb) - облегченная версия решения, основанного на анализе компонент связности.

## Зависимости

- `numpy`
- `opencv` (`cv2`)

## Проверка решения

Для проверки решения откройте файл [`ConnectedComponents.ipynb`](./ConnectedComponents.ipynb) и скопируйте оттуда функцию `get_mask`. Также импортируйте все функции из файлов `utils_uint8.py` и `binarization.py`. Маска выдаётся в формате "0 - не выделенный пиксель, 255 - выделенный", тип данных - `numpy.uint8`.
