#!/usr/bin/env python3
"""
Parkinson's Disease Detection with XGBoost + KFold CV + Feature Selection + Styled Tables + Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Загрузка датасета
data = pd.read_csv("parkinsons.data")  # путь к файлу
X = data.drop(columns=['name', 'status'])
y = data['status']

# 2. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Базовая модель для оценки важности признаков
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

# 5. Отбор топ-15 признаков
importances = base_model.feature_importances_
top_indices = importances.argsort()[::-1][:15]
X_train_selected = X_train[:, top_indices]
X_test_selected = X_test[:, top_indices]
top_features = X.columns[top_indices].tolist()
print("Используем признаки:", top_features)

# 6. Финальная модель
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

# 7. Кросс-валидация
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(final_model, X_train_selected, y_train, cv=kf, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 8. Обучение на всей train выборке и проверка на test
final_model.fit(X_train_selected, y_train)
y_pred = final_model.predict(X_test_selected)
test_acc = accuracy_score(y_test, y_pred)
print(f"Точность на тестовой выборке: {test_acc:.4f}")

# 9. Визуализация важности признаков
plt.figure(figsize=(10,6))
sns.barplot(x=importances[top_indices], y=top_features, palette='viridis')
plt.title("Feature Importance (Top 15)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 10. Матрица ошибок в виде таблицы и heatmap
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Healthy','PD'], columns=['Healthy','PD'])
display(cm_df.style.background_gradient(cmap='Blues').format("{:.0f}"))

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy','PD'], yticklabels=['Healthy','PD'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Test Accuracy={test_acc:.2%})")
plt.tight_layout()
plt.show()

# 11. Classification report в виде таблицы
report_dict = classification_report(y_test, y_pred, target_names=['Healthy','PD'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
display(report_df.style.background_gradient(cmap='Blues', subset=['precision','recall','f1-score']).format("{:.2f}"))

# 12. Распределение предсказаний
plt.figure(figsize=(6,4))
sns.countplot(x=y_pred, palette='Set2')
plt.xticks([0,1], ['Healthy','PD'])
plt.title("Distribution of Predictions on Test Set")
plt.show()

