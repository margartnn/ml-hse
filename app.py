#я прямо куски копировала из ноутбука про streamlit, с комментариями оттуда же

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="HW1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource  # Кэшируем модель (загружается только один раз)
def load_model():
    with open('models/model_ridge.pkl', 'rb') as f:
        model_ridge = pickle.load(f)
    model = model_ridge["model"]
    ridge_scaler = model_ridge["scaler"]
    feature_names = model_ridge["feature_names"]
    return model, ridge_scaler, feature_names

model, ridge_scaler, feature_names = load_model()

#тут общие приготовления из части1
def prepare_eda(df_original):
    """Приводим данные к формату обучения модели для EDA"""
    df = df_original.copy()
    df = df.drop(columns=['name'], errors='ignore')
    df = df.drop(columns=['torque'], errors='ignore')
    df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
    for i in ['mileage', 'engine', 'max_power']:
        df[i] = pd.to_numeric(df[i].str.extract('(\d+\.?\d*)')[0], downcast='float', errors='coerce')
    for i in ['mileage', 'engine', 'max_power', 'seats']:
        median = df[i].median()
        df[i].fillna(median, inplace=True)
    median_mileage = df['mileage'].median()
    df['mileage'] = df['mileage'].replace(0, median_mileage)
    median_max_power = df['max_power'].median()
    df['max_power'] = df['max_power'].replace(0, median_max_power)
    return df

def prepare_features(df_original):
    """Приводим данные к формату обучения модели"""
    df = df_original.copy()
 #  df = df.drop(columns=['selling_price'], errors='ignore')
    df = df.drop(columns=['name'], errors='ignore')
    df = df.drop(columns=['torque'], errors='ignore')
    df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
    for i in ['mileage', 'engine', 'max_power']:
        df[i] = pd.to_numeric(df[i].str.extract('(\d+\.?\d*)')[0], downcast='float', errors='coerce')
    for i in ['mileage', 'engine', 'max_power', 'seats']:
        median = df[i].median()
        df[i].fillna(median, inplace=True)
    median_mileage = df['mileage'].median()
    df['mileage'] = df['mileage'].replace(0, median_mileage)
    median_max_power = df['max_power'].median()
    df['max_power'] = df['max_power'].replace(0, median_max_power)
    categorial = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
    df = pd.get_dummies(df, columns=categorial, drop_first=True)
    df_encoded = df.reindex(columns=feature_names, fill_value=0)
    return df_encoded

@st.cache_data  # Кэшируем загруженные данные
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# В интерфейсе:
uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    df_eda = prepare_eda(df)
    X_ridge = prepare_features(df)
    st.subheader("Первые 5 строк датасета") #датасет c ЦЕНОЙ специально вывела, чтобы сравнить со следующим блоком было можно цены 
    st.dataframe(df_eda.head()) #я не перепутала выводимые датасеты, такая задумка! 

    # Предсказание
    predictions = model.predict(X_ridge)
    
    # Визуализация результатов
    st.subheader("Первые 5 предсказаний Ridge:")
    st.write(predictions[:5])

    if 'selling_price' in df.columns:
        y_true = df_eda['selling_price']

        ridge_r2 = r2_score(y_true, predictions)
        ridge_mse = mean_squared_error(y_true, predictions)

        st.header("Основные метрики Ridge")
        st.write(f"**R2:** {ridge_r2}")
        st.write(f"**MSE:** {ridge_mse}")

        st.header("Веса модели")
        if hasattr(model, "best_estimator_"):
            coef = model.best_estimator_.coef_
        else:
            coef = model.coef_
        coef_series = pd.Series(coef, index=feature_names)
        coef_sorted = coef_series.sort_values()
        st.write(coef_sorted)

        st.header("Корреляции между признаками")

        numeric_df = df_eda[['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']]
        corr_matrix = numeric_df.corr()
        st.subheader("Попарные корреляции")
        st.dataframe(corr_matrix)

        target_corr = corr_matrix['selling_price'].drop('selling_price')
        strongest = target_corr.idxmax(), target_corr.max()
        weakest = target_corr.abs().idxmin(), target_corr[target_corr.abs().idxmin()]

        st.write(f"**Самая сильная с целевой переменной** {strongest[0]} = {strongest[1]}")
        st.write(f"**Самая слабая с целевой переменной** {weakest[0]} = {weakest[1]}")

        st.subheader("Pairplot по числовым признакам")
        cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        pairplot_fig = sns.pairplot(numeric_df)
        st.pyplot(pairplot_fig)

       

        