## Комментарии по коду примеров моделей машинного обучения

Представленные модели (кроме logistic_classification) годяться как для задачи классификации так и для регресии.

Для простоты я пытался каждую из моделей рассматривать в контексте одной задачи (классификации, регресии) и на одной выборке (breast_cancer, boston). Исключение: KNN-модель (модель ближайших соседей), которую я тренировал как для классификации так и для регресии.

Если выборка была из пакета breast_cancer, то рассматривалась задача классификации.

Если выборка была из пакета boston, то рассматривалась задача регресии.

Модуль extra.py содержит дополнительные функции (вывод наиболее важных признаков тренированной модели, например).