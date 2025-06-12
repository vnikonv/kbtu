# 🚀 Anti-Fraud Detection App (Streamlit)

Интерактивное веб-приложение на Streamlit для обучения, валидации и использования антифрод-моделей (Logistic Regression и XGBoost) на основе кредитных данных заемщиков.

---

## 📌 Возможности

- Загрузка пользовательского датасета (CSV, Excel)
- Обработка пропусков, кодирование и масштабирование признаков
- Обучение моделей: Logistic Regression и XGBoost
- Расчет метрик:
  - Gini
  - KS-индекс
  - Accuracy, Precision, Recall
- Визуализация ROC-кривой и распределения вероятностей
- Сохранение модели и предсказаний
- Проверка новых данных через интерфейс

---

## 🔧 Установка

```bash
git clone https://github.com/your-username/anti-fraud-streamlit.git
cd anti-fraud-streamlit
pip install -r requirements.txt
```

---

## 🚀 Запуск

```bash
streamlit run app.py
```

---

## 🧭 Как пользоваться

1. Перейдите во вкладку **"🏋️‍♂️ Обучение XGBoost"** или **"📊 Обучение Logistic Regression"**
2. Загрузите датасет в формате `.csv` или `.xlsx` с колонкой `GB_flag` как таргет
3. Дождитесь завершения обучения модели и отображения метрик
4. При желании скачайте файл с предсказаниями на тестовой выборке
5. Перейдите во вкладку **"🔍 Проверка XGBoost"** или **"📊 Проверка Logistic Regression"**
6. Загрузите новые данные для оценки — вы получите вероятности мошенничества

---

## 📂 Структура проекта

```
📁 anti-fraud-streamlit/
│
├── app.py                  # Основной файл Streamlit-приложения
├── model_xgb.json          # Модель XGBoost
├── model_lr.json           # Модель Logistic Regression
├── scaler.json             # Масштабатор
├── label_encoders.json     # Энкодеры категориальных признаков
├── features.json           # Использованные признаки
├── test_predictions.csv    # Предсказания
├── requirements.txt        # Зависимости проекта
└── ...
```

---

## 📥 Входные данные

Таблица с признаками заемщиков и целевым признаком `GB_flag` (1 — мошенник, 0 — нет).

---

## 📈 Метрики

- **Gini**: \(2 	imes AUC - 1\)
- **KS-индекс**: максимум разности между распределениями плохих и хороших
- **Accuracy, Precision, Recall** — стандартные метрики классификации

## QR КОД
<img width="366" alt="Снимок экрана 2025-03-25 в 22 30 17" src="https://github.com/user-attachments/assets/e37f73ce-f8d2-4a43-944e-e8e5d9e0a179" />