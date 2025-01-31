import json
import streamlit as st
import pandas as pd
from pycaret.clustering import setup, create_model, assign_model, save_model, load_model, predict_model
import plotly.express as px

# Ścieżki do plików
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'

# Wczytywanie danych (bez użycia cache)
def load_data():
    return pd.read_csv(DATA, sep=';')

# Funkcja do trenowania modelu (bez cache)
def train_and_save_model(df, model_name: str):
    clustering_setup = setup(df, session_id=123)
    model = create_model('kmeans', num_clusters=7)
    df_with_clusters = assign_model(model)
    save_model(model, model_name)
    return df_with_clusters, model

# 🟢 Funkcja do poprawnego generowania nazwy klastra
def generate_cluster_name(cluster_df):
    if cluster_df.empty:
        return "Nieokreślona grupa"

    most_common_edu = cluster_df["edu_level"].mode()[0].lower()
    most_common_place = cluster_df["fav_place"].mode()[0].lower()
    most_common_animal = cluster_df["fav_animals"].mode()[0].lower()

    return f"Osoby lubiące przebywać {most_common_place} i kochające {most_common_animal}, posiadające {most_common_edu} wykształcenie"

# 🟢 Funkcja do generowania opisu klastra z poprawną gramatyką
def generate_cluster_description(cluster_df):
    if cluster_df.empty:
        return "Brak dostępnych informacji o tej grupie."

    most_common_edu = cluster_df["edu_level"].mode()[0].lower()
    most_common_place = cluster_df["fav_place"].mode()[0].lower()
    most_common_animal = cluster_df["fav_animals"].mode()[0].lower()
    most_common_age = cluster_df["age"].mode()[0]
    most_common_gender = cluster_df["gender"].mode()[0].lower()

    gender_text = "mężczyzn" if most_common_gender == "mężczyzna" else "kobiet"

    return (f"W tym klastrze znajdują się osoby, które lubią przebywać {most_common_place}, "
            f"kochają {most_common_animal} i posiadają {most_common_edu} wykształcenie. "
            f"Większość z nich to {gender_text} w wieku {most_common_age}.")

# Ładowanie danych i trenowanie modelu
df = load_data()
df_with_clusters, model = train_and_save_model(df, MODEL_NAME)
# generate_cluster_name(cluster_df)

# Funkcja do przygotowywania danych użytkownika
def prepare_person_df(age, edu_level, fav_animals, fav_place, gender):
    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender
    }])
    return person_df

# Użytkownik podaje swoje dane
with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

# Przycisk do wykonania akcji
if st.button("Sprawdź klaster"):
    person_df = prepare_person_df(age, edu_level, fav_animals, fav_place, gender)
    cluster_prediction = predict_model(model, data=person_df)
    predicted_cluster_id = cluster_prediction["Cluster"].values[0]

    # Pobranie użytkowników w danym klastrze
    same_cluster_df = df_with_clusters[df_with_clusters["Cluster"] == predicted_cluster_id]

    # Generowanie poprawnej nazwy i opisu klastra
    cluster_name = generate_cluster_name(same_cluster_df)
    cluster_description = generate_cluster_description(same_cluster_df)

    # 🟢 Wyświetlanie wyników
    st.header(f"Najbliżej Ci do grupy: {cluster_name}")
    st.markdown(cluster_description)

    # Liczba osób w tym samym klastrze
    st.metric("Liczba twoich znajomych", len(same_cluster_df))

    # Wizualizacje danych
    fig = px.histogram(same_cluster_df, x="age", title="Rozkład wieku w grupie")
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df, x="edu_level", title="Rozkład wykształcenia w grupie")
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df, x="fav_animals", title="Rozkład ulubionych zwierząt w grupie")
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df, x="fav_place", title="Rozkład ulubionych miejsc w grupie")
    st.plotly_chart(fig)

    fig = px.histogram(same_cluster_df, x="gender", title="Rozkład płci w grupie")
    st.plotly_chart(fig)

