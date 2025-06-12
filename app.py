import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

st.set_page_config(page_title='Antifraud AI', page_icon = "ammit_search.png", layout = 'wide', initial_sidebar_state = 'auto')

st.markdown("""
<style>
/* Общий стиль сайдбара */
section[data-testid="stSidebar"] {
    background-color: #02b871 !important;
    color: white !important;
    padding-top: 10px !important;
}

/* Надписи и тексты в сайдбаре */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span {
    color: white !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border-radius: 12px !important;
}

/* Селектбокс и поля ввода */
section[data-testid="stSidebar"] div[role="combobox"],
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background-color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    box-shadow: none !important;
}

/* SVG иконки в сайдбаре */
section[data-testid="stSidebar"] svg {
    color: white !important;
}

/* Выпадающие списки и опции */
ul[role="listbox"],
li[role="option"] {
    background-color: #000000 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
}

li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #000000 !important;
}

/* Устранение стандартных границ и теней от BaseWeb */
[data-baseweb="select"],
[data-baseweb="select"] * {
    background-color: #000000 !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}

img {
    border-radius: 32px !important;
}

label div p {
    font-size: 24px !important;
    font-weight: 600 !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 5% !important;
}

</style>
""", unsafe_allow_html=True)





# Пути для сохранения моделей и параметров
MODEL_PATH_XGB = "model_xgb.json"
MODEL_PATH_LR = "model_lr.json"
LABEL_ENCODERS_PATH = "label_encoders.json"
FEATURES_PATH = "features.json"
SCALER_PATH = "scaler.json"

# Выбор вкладки через selectbox
menu = st.sidebar.selectbox(
    "Выберите вкладку:",
    [
        "🏋️‍♂️ Обучение XGBoost",
        "📊 Обучение Logistic Regression",
        "🔍 Проверка XGBoost",
        "📊 Проверка Logistic Regression"
    ]
)

# Выбор типа модели
if "XGBoost" in menu:
    model_type = "XGBoost" if "Обучение" in menu else "xgb"
elif "Logistic Regression" in menu:
    model_type = "Logistic Regression" if "Обучение" in menu else "lr"

# Функции

def save_model(model, feature_names, model_path):
    if isinstance(model, xgb.XGBClassifier):
        model.save_model(model_path)
    else:
        with open(model_path, "w") as f:
            json.dump({"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "features": feature_names}, f)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)

def load_model(model_path, model_type):
    if model_type == "xgb":
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        with open(model_path, "r") as f:
            data = json.load(f)
        model = LogisticRegression()
        model.coef_ = np.array(data["coef"])
        model.intercept_ = np.array(data["intercept"])
        model.classes_ = np.array([0, 1])
    return model

def save_label_encoders(label_encoders):
    with open(LABEL_ENCODERS_PATH, "w") as f:
        json.dump({col: le.classes_.tolist() for col, le in label_encoders.items()}, f)

def load_label_encoders():
    with open(LABEL_ENCODERS_PATH, "r") as f:
        return {col: LabelEncoder().fit(classes) for col, classes in json.load(f).items()}

def save_scaler(scaler):
    with open(SCALER_PATH, "w") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}, f)

def load_scaler():
    with open(SCALER_PATH, "r") as f:
        data = json.load(f)
    scaler = StandardScaler()
    scaler.mean_ = np.array(data["mean"])
    scaler.scale_ = np.array(data["scale"])
    return scaler

def load_feature_names():
    with open(FEATURES_PATH, "r") as f:
        return json.load(f)

def encode_categorical(df):
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def align_features(df, feature_names):
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

# Интерфейс
icon = "bcc_icon.png"

left_co, cent_co, last_co = st.columns([0.35, 0.3, 0.35])
with cent_co:
    st.image("ammit.png")
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(icon, width=100)
with col2:
    st.title("Антифрод ML")

if "Обучение" in menu:
    st.header(f"📌 Обучение модели {model_type}")
    uploaded_file = st.file_uploader("Загрузите CSV или Excel-файл", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        if "GB_flag" not in df.columns:
            st.error("❌ Ошибка: В файле нет столбца 'GB_flag'")
        else:
            X, y = df.drop(columns=["GB_flag"]), df["GB_flag"]
            X_raw = X.copy()
            X, label_encoders = encode_categorical(X)
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_original_for_export = X.copy()
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
            _, X_test_raw = train_test_split(X_raw, test_size=0.3, random_state=42)
            feature_names = list(X.columns)
            model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric="logloss", n_jobs=-1) if model_type == "XGBoost" else LogisticRegression()
            model.fit(X_train, y_train)
            # Обучение модели с замером времени
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Предсказания
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]


            # Индекс Колмогорова-Смирнова
            def ks_statistic(y_true, y_proba):
                from scipy.stats import ks_2samp
                return ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic

            # Метрики (тест)
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred)
            roc_auc = roc_auc_score(y_test, y_test_proba)
            gini = 2 * roc_auc - 1
            ks = ks_statistic(y_test.to_numpy(), y_test_proba)

            # Метрики (обучение)
            accuracy_train = accuracy_score(y_train, y_train_pred)
            precision_train = precision_score(y_train, y_train_pred)
            recall_train = recall_score(y_train, y_train_pred)
            roc_auc_train = roc_auc_score(y_train, y_train_proba)
            gini_train = 2 * roc_auc - 1
            ks_train = ks_statistic(y_train.to_numpy(), y_train_proba)

            # Сохраняем всё
            model_path = MODEL_PATH_XGB if model_type == "XGBoost" else MODEL_PATH_LR
            save_model(model, feature_names, model_path)
            save_label_encoders(label_encoders)
            save_scaler(scaler)

            # Отображение результатов
            st.success(f"✅ {model_type} модель обучена и сохранена")

            st.subheader("📈 Результаты обучения (тест)")
            st.markdown(f"⏱ **Время обучения:** {train_time:.2f} сек")
            st.markdown(f"📊 **Gini индекс:** {gini:.4f}")
            st.markdown(f"📊 **KS индекс:** {ks:.4f}")
            st.markdown(f"✅ **Accuracy:** {accuracy:.4f}")
            st.markdown(f"🎯 **Precision:** {precision:.4f}")
            st.markdown(f"🔁 **Recall:** {recall:.4f}")

            st.subheader("📈 Результаты обучения (обучение)")
            st.markdown(f"⏱ **Время обучения:** {train_time:.2f} сек")
            st.markdown(f"📊 **Gini индекс:** {gini_train:.4f}")
            st.markdown(f"📊 **KS индекс:** {ks_train:.4f}")
            st.markdown(f"✅ **Accuracy:** {accuracy_train:.4f}")
            st.markdown(f"🎯 **Precision:** {precision_train:.4f}")
            st.markdown(f"🔁 **Recall:** {recall_train:.4f}")

            # Дополнительные данные
            st.subheader("🔢 Пример расчётов")
            # Таблица метрик
            metrics_df = pd.DataFrame({
                "Метрика": ["Время обучения (сек)", "Gini", "KS", "Accuracy", "Precision", "Recall", "ROC AUC"],
                "Значение": [round(train_time, 2), round(gini, 4), round(ks, 4), round(accuracy, 4),
                             round(precision, 4), round(recall, 4), round(roc_auc, 4)]
            })
            metrics_df_train = pd.DataFrame({
                "Метрика": ["Время обучения (сек)", "Gini", "KS", "Accuracy", "Precision", "Recall", "ROC AUC"],
                "Значение": [round(train_time, 2), round(gini_train, 4), round(ks_train, 4), round(accuracy_train, 4),
                             round(precision_train, 4), round(recall_train, 4), round(roc_auc_train, 4)]
            })
            st.subheader("📊 Сводная таблица метрик (тест)")
            st.table(metrics_df)

            st.subheader("📊 Сводная таблица метрик (обучение)")
            st.table(metrics_df_train)

            # ROC-кривая
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            fig_roc = plt.figure()
            plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC-кривая")
            plt.legend()
            st.subheader("📈 ROC-кривая")
            st.pyplot(fig_roc)

            # Гистограммы распределения вероятностей
            fig_hist = plt.figure()
            plt.hist(y_test_proba[y_test == 0], bins=30, alpha=0.6, label="Класс 0 (не мошенник)")
            plt.hist(y_test_proba[y_test == 1], bins=30, alpha=0.6, label="Класс 1 (мошенник)")
            plt.title("📊 Распределение вероятностей по классам")
            plt.xlabel("Предсказанная вероятность")
            plt.ylabel("Количество")
            plt.legend()
            st.subheader("🔍 Распределение вероятностей")
            st.pyplot(fig_hist)
            st.markdown(f"**Рассчитанный балл (пример):** {y_test_pred[0]}")
            st.markdown(f"**Вероятностная оценка (пример):** {y_test_proba[0]:.4f}")

            # Использованные переменные
            st.subheader("📌 Использованные переменные")
            st.code(", ".join(feature_names))

            # Таргет и принадлежность к выборке
            subset_info = pd.DataFrame({
                "target": pd.concat([y_train, y_test], ignore_index=True),
                "subset": ["train"] * len(y_train) + ["test"] * len(y_test)
            })
            st.subheader("🧪 Разметка выборок")
            st.dataframe(subset_info.sample(10, random_state=42))
            # Сохраняем тестовые результаты в DataFrame
            # Восстанавливаем X_test до масштабирования
            X_test_original = X_test_raw.copy()


            # Финальный датафрейм с нужной инфой
            # Создаём финальный DataFrame из оригинальных данных
            results_df = X_test_raw.copy()
            results_df["Предсказанный класс"] = y_test_pred
            results_df["Вероятность"] = y_test_proba
            results_df["Истинный GB_flag"] = y_test.values

            # Кнопка для скачивания
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать результаты на тестовой выборке (оригинальные значения)",
                data=csv,
                file_name="test_predictions.csv",
                mime="text/csv"
            )

            st.success(f"✅ {model_type} модель сохранена")

elif "Проверка" in menu:
    model_path = MODEL_PATH_XGB if model_type == "xgb" else MODEL_PATH_LR
    if os.path.exists(model_path):
        model = load_model(model_path, model_type)
        scaler = load_scaler()
        feature_names = load_feature_names()
        uploaded_file = st.file_uploader("Загрузите тестовый файл", type=["csv", "xlsx", "xls"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            df_original = df.copy()

            # Выбор переменных с помощью флажков
            selected_features = st.sidebar.multiselect("🔧 Отметьте переменные для отображения:", df.columns.tolist(),
                                                       default=df.columns.tolist())

            df_aligned = align_features(df, feature_names)
            imputer = SimpleImputer(strategy="mean")
            df_imputed = pd.DataFrame(imputer.fit_transform(df_aligned), columns=df_aligned.columns)
            df_scaled = pd.DataFrame(scaler.transform(df_imputed), columns=df_imputed.columns)

            predictions = model.predict_proba(df_scaled)[:, 1]
            df_original["Вероятность мошенничества"] = predictions

            # Ползунок для отображения количества строк
            st.subheader("📋 Результаты проверки")
            rows_to_display = st.slider("Количество отображаемых строк", min_value=1,
                                        max_value=min(100, len(df_original)), value=min(10, len(df_original)), step=1)

            # Итоговая таблица
            st.dataframe(df_original[selected_features + ["Вероятность мошенничества"]].head(rows_to_display))
    else:
        st.error("❌ Сначала обучите модель")
