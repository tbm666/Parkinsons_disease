Parkinson’s Disease Detection with XGBoost

📌 Описание

Этот проект демонстрирует использование машинного обучения для ранней диагностики болезни Паркинсона на основе голосовых данных. Мы применяем алгоритм XGBoost с отбором признаков и кросс-валидацией для достижения высокой точности.

📊 Датасет

Источник: UCI Parkinson’s Disease Dataset

Размер: 195 записей, 23 признака

Целевая переменная: status (0 — здоров, 1 — болезнь Паркинсона)

⚙️ Структура проекта
Parkinsons-Disease-Detection/
│
├── data/
│   └── parkinsons.data  # Исходный CSV файл
│
├── src/
│   └── main.py          # Основной скрипт обучения и оценки модели
│
├── requirements.txt     # Зависимости проекта
└── README.md            # Этот файл

🛠 Установка

Клонируйте репозиторий:

git clone https://github.com/yourusername/Parkinsons-Disease-Detection.git
cd Parkinsons-Disease-Detection


Установите зависимости:

pip install -r requirements.txt

▶️ Запуск
python src/main.py

📈 Результаты

Пример вывода:

Используемые признаки: ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Jitter(%)', ...]
CV Accuracy: 0.9450 ± 0.0321
Точность на тестовой выборке: 0.9231

📚 Литература

UCI Parkinson’s Disease Dataset

XGBoost Documentation
