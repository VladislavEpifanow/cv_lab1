#  Задача выделения бинарной маски переднего плана
## Датасет
Расположен в архиве `Lab1.zip`

Включает в себя:
* 100 изображений кошек и собак 
* 100 тринарных масок для данных изображений (в ходе решения задачи потребуется перевести тринарные маски в бинарные, пример перевода в файле `metric_evaluation.py`)

Данные отобраны из датасета [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Описание задачи
* Язык программирования - `Python`
* Выделение бинарной маски изображения может быть выполнено с помощью различных математических операций (самописных или готовых решений из сторонних библиотек), без использования методов машинного обучения

* В файле `solution_example.py` реализовать функцию `get_foreground_mask()`

## Метрики
**Метрика точности**
* В качестве метрики оценки результата используется IoU (Intersection over Union), расчитанная для класса foreground
* Вычисление метрики выполняется последовательно для каждого изображения в файле `metric_evaluation.py` в функции `evaluate_iou()`

![image_2023-09-17_20-47-39](https://learnopencv.com/wp-content/uploads/2022/12/feature-image-iou-1-1024x292.jpg)

**Скорость выполнения**
* Измерить скорость выполнения в с/Мп (секунда на мегапиксель)


## Baseline
* IoU вычисляется для каждого изображения, затем определяется среднее значение
* _Нижний порог метрики_ равен `53.56%` и достигается, если в качестве решения передавать все координаты изображения
* При помощи решения составителей задания было получено значение метрики, равное `88.81%`

* Скорость выполнения решения составителей задания, исполняемого на Intel(R) Xeon(R) CPU @ 2.20GHz, RAM 12Gi, составила `6.65 с/Мп`

## Алгоритм запуска 
**Локально:**
  * Распаковать содержимое архива `Lab1.zip` внутри папки `lab1`
  * `pip install -r requirements.txt`
  * В корне проекта выполнить `python -m lab1`

**Colab:**
* Использовать данный [ноутбук](https://colab.research.google.com/drive/1_UEDcEHm3FgnuMdte1ll98bF5MBQ2sjI?usp=sharing)


## Решение
* В качестве основы для решения выделения бинарной маски переднего вида была использована адаптивная бинаризация и использованием определения порога по Гауссу
* Зачастую, после бинаризации появляется проблема, что не всегда понятно, какие пиксели были выделены как изображение, а какие как фон ( например, возможна ситуация фон выделяется белым, а объект черным, а также наоборот)
* Для решения данной проблемы используется расстояние Евклида, т.е. пробегаемся по всем пикселям в двух группах бикселей ( белых и черных), считаем расстояние Евклида до центра, суммируем на каждом объекте в группе и делим на их количество
* Соответственно, чем меньше метрика, тем ближе группа пикселей к середине, а из данных видно, что объект как правило находится ближе к центру.
  
