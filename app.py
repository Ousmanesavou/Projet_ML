# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 08:48:00 2024

@author: hp
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Configuration de la page avec titre, icône et disposition
st.set_page_config(
    page_title="Prédiction de l'abandon des clients", 
    page_icon="📊", 
    layout="wide"
)

# Bannière d'accueil
st.markdown(
    """
    <div style="background-color:#0d6efd;padding:10px;border-radius:10px;text-align:center;">
    <h1 style="color:white;">Prédiction de l'Abandon des Clients 📉</h1>
    <p style="color:white;">Bienvenue dans cette application interactive pour la prédiction du churn. Explorez, analysez, et prédisez l'abandon de vos clients avec des modèles de machine learning.</p>
    </div>
    <br>
    """, unsafe_allow_html=True)

# Logo de l'application (facultatif)
st.sidebar.image(r"C:\Users\hp\Projet_ML\logo_orange.png", use_column_width=True)

# Barre latérale pour le chargement du fichier et la configuration
st.sidebar.title("🔧 Configuration")
uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV ici", type="csv")

# Section pour les paramètres de prétraitement
st.sidebar.subheader("📂 Paramètres de Prétraitement")

# Variables pour la session
if 'df' not in st.session_state:
    st.session_state.df = None

# Vérification du fichier chargé
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

# Affichage des données si le fichier est chargé
if st.session_state.df is not None:
    df = st.session_state.df
    st.write("### Aperçu du jeu de données téléchargé :")
    st.dataframe(df.head())

    # Affichage des informations du jeu de données
    with st.expander("Voir plus de détails sur les colonnes :"):
        st.write(df.describe())
        st.write("Types de données :", df.dtypes)

    # Prétraitement des données
    if st.sidebar.button("Prétraiter les Données"):
        if 'Surname' in df.columns:
            df = df.drop(['Surname'], axis=1)  # Suppression de la colonne inutile

        # Traitement des valeurs manquantes
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)

        # Encodage des colonnes catégorielles
        df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

        # Normalisation des colonnes numériques
        scaler = StandardScaler()
        columns_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

        # Stockage du dataframe prétraité
        st.session_state.df_preprocessed = df
        st.success("Les données ont été prétraitées avec succès.")

# Sélection des fonctionnalités de l'application
st.sidebar.title("🚀 Sélection des Fonctionnalités")
feature_selection = st.sidebar.radio(
    "Choisissez l'action à effectuer :", 
    ["Visualisation des Données", "Entraîner le Modèle", "Évaluation du Modèle"]
)

# Visualisation des données
if feature_selection == "Visualisation des Données":
    st.subheader("Visualisation des Données 📊")
    if 'df_preprocessed' in st.session_state:
        with st.expander("Afficher la Matrice de Corrélation"):
            corr = st.session_state.df_preprocessed.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Distribution des variables
        with st.expander("Afficher la Distribution des Variables"):
            numerical_columns = st.session_state.df_preprocessed.select_dtypes(include=[np.number]).columns
            column = st.selectbox("Choisissez une variable numérique :", numerical_columns)
            fig = px.histogram(st.session_state.df_preprocessed, x=column, nbins=30, title=f"Distribution de {column}")
            st.plotly_chart(fig)

# Entraînement du modèle
elif feature_selection == "Entraîner le Modèle":
    st.subheader("Entraîner le Modèle 🎯")
    if 'df_preprocessed' in st.session_state:
        df = st.session_state.df_preprocessed
        X = df.drop(['Exited'], axis=1)
        y = df['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Paramètres du modèle ajustables
        st.markdown("### Paramètres du Modèle")
        max_iter = st.slider("Nombre d'itérations pour le modèle MLP", 100, 1000, step=100)

        st.write("**Entraînement des modèles en cours...**")
        with st.spinner("Entraînement en cours..."):
            # Modèles avec hyperparamètres personnalisés
            log_model = LogisticRegression(max_iter=1000)
            rf_model = RandomForestClassifier()
            mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=max_iter)

            # Entraîner les modèles
            log_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            mlp_model.fit(X_train, y_train)

        st.success("Les modèles ont été entraînés avec succès.")
        st.session_state.models = {"Logistic Regression": log_model, 
                                   "Random Forest": rf_model, 
                                   "Neural Network": mlp_model}

# Évaluation du modèle
elif feature_selection == "Évaluation du Modèle":
    st.subheader("Évaluation des Modèles 📈")
    if 'models' in st.session_state:
        model_choice = st.selectbox("Sélectionnez le modèle à évaluer", st.session_state.models.keys())
        model = st.session_state.models[model_choice]

        # Prédictions et métriques
        X = st.session_state.df_preprocessed.drop(['Exited'], axis=1)
        y = st.session_state.df_preprocessed['Exited']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = model.predict(X_test)

        # Affichage des métriques
        st.write(f"**Performance de {model_choice}**")
        st.write(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"Précision : {precision_score(y_test, y_pred):.2f}")
        st.write(f"Rappel : {recall_score(y_test, y_pred):.2f}")
        st.write(f"F1-Score : {f1_score(y_test, y_pred):.2f}")

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Matrice de Confusion :")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Churn", "Churn"], yticklabels=["Non-Churn", "Churn"], ax=ax)
        st.pyplot(fig)

# Fin de l'application
st.write("Fin de l'application.")
