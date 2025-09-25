# Parkinson’s Disease Detection with XGBoost

![Parkinson’s Disease Detection](https://data-flair.training/blogs/wp-content/uploads/sites/2/2019/06/Parkinson-Disease-Detection-Using-XGBoost.jpg)

## 📌 Описание

Проект демонстрирует использование машинного обучения для ранней диагностики болезни Паркинсона на основе голосовых данных. Используется алгоритм **XGBoost** с отбором признаков и **кросс-валидацией** для оценки качества модели.

## 📊 Датасет

* **Источник**: [UCI Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
* **Размер**: 195 записей, 23 признака
* **Целевая переменная**: `status` (0 — здоров, 1 — болезнь Паркинсона)

## ⚙️ Структура проекта

```
Parkinsons_disease/
├── parkinsons.data      # CSV с исходными данными
├── parkinsons.py        # Основной скрипт обучения модели
├── requirements.txt     # Зависимости проекта
└── README.md            # Этот файл
```

## 🛠 Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/tbm666/Parkinsons_disease.git
cd Parkinsons_disease
```

2. Установите зависимости:

```bash
pip install -r requirements.txt
```

## ▶️ Запуск

```bash
python parkinsons.py
```

## 📈 Результаты

Пример вывода:

```
Используемые признаки: ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Jitter(%)', ...]
CV Accuracy: 0.9450 ± 0.0321
Точность на тестовой выборке: 0.9231
```

## 📚 Литература

* [UCI Parkinson’s Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)

