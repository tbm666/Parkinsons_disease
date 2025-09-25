#!/usr/bin/env python3
"""
Parkinson's Disease Detection with XGBoost + KFold CV + Feature Selection
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Загрузка датасета
data = pd.read_csv("parkinsons.data")  # путь к файлу
X = data.drop(columns=['name', 'status'])
y = data['status']

# 2. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Разделение на train/test для финальной проверки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Обучаем базовую модель для оценки важности признаков
base_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
base_model.fit(X_train, y_train)

# 5. Отбор признаков: берем топ-15 по важности
importances = base_model.feature_importances_
top_indices = importances.argsort()[::-1][:15]
X_train_selected = X_train[:, top_indices]
X_test_selected = X_test[:, top_indices]

print("Используем признаки:", X.columns[top_indices].tolist())

# 6. Финальная модель на выбранных признаках
final_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# 7. Кросс-валидация на train
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_train_selected, y_train, cv=kf, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 8. Обучение на всей train выборке и проверка на test
final_model.fit(X_train_selected, y_train)
y_pred = final_model.predict(X_test_selected)
test_acc = accuracy_score(y_test, y_pred)
print(f"Точность на тестовой выборке: {test_acc:.4f}")