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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Fonction pour l'entraînement des modèles
def train_models(X_train, y_train):
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
    mlp_model.fit(X_train, y_train)

    return log_model, rf_model, mlp_model

# Fonction pour évaluer les modèles
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

# Titre et description de l'application
st.title("Prédiction de l'abandon des clients")
st.markdown("""
    Bienvenue dans l'application de prédiction du churn. 
    Téléchargez vos données, choisissez un modèle, et obtenez des résultats avec des visualisations.
""")

# Sélection du jeu de données via l'interface
uploaded_file = st.file_uploader("Télécharger votre fichier CSV", type="csv")
if uploaded_file is not None:
    # Charger les données et afficher un aperçu
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(df.head())
    st.write("Types de données :")
    st.write(df.dtypes)

    # Bouton de prétraitement des données
    if st.button("Prétraiter les données"):
        df = df.drop(['Surname'], axis=1, errors='ignore')

        # Conversion des colonnes en numérique (avec gestion des erreurs)
        columns_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        for col in columns_to_scale:
            # Supprimer les caractères non numériques
            df[col] = df[col].replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir en numérique, remplacer les erreurs par NaN

        # Suppression des lignes contenant des NaN dans les colonnes à scaler
        df.dropna(subset=columns_to_scale, inplace=True)

        # Vérification si le DataFrame n'est pas vide après suppression
        if df.empty:
            st.error("Le DataFrame est vide après le prétraitement. Vérifiez vos données.")
        else:
            # Encodage des variables catégorielles
            df = pd.get_dummies(df, drop_first=True)

            # Normalisation des données numériques
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numerical_cols) == 0:
                st.error("Aucune colonne numérique trouvée pour la normalisation.")
            else:
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                st.success("Les données ont été prétraitées avec succès.")

    # Sélection de la fonctionnalité à exécuter
    action = st.selectbox("Choisissez l'action à effectuer", 
                          ["Entraîner le modèle", "Visualisation des données", "Évaluation du modèle"])

    # Séparation des données en variables X et y
    if st.button("Séparer les données"):
        if 'Exited' not in df.columns:
            st.error("La colonne 'Exited' n'est pas présente dans le DataFrame.")
        else:
            X = df.drop(['Exited'], axis=1)
            y = df['Exited']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success("Les données ont été séparées en variables indépendantes et dépendantes.")
    
    if action == "Entraîner le modèle":
        if st.button("Entraîner les modèles"):
            if 'X_train' in st.session_state and 'y_train' in st.session_state:
                st.session_state.log_model, st.session_state.rf_model, st.session_state.mlp_model = train_models(st.session_state.X_train, st.session_state.y_train)
                st.success("Les modèles ont été entraînés avec succès.")
            else:
                st.error("Les données d'entraînement ne sont pas disponibles.")

    # Évaluation du modèle
    if action == "Évaluation du modèle":
        st.subheader("Choix du modèle pour l'évaluation")
        model_choice = st.selectbox("Choisissez un modèle à évaluer", 
                                    ["Régression Logistique", "Random Forest", "Réseau de Neurones"])

        if st.button("Évaluer le modèle"):
            if 'rf_model' in st.session_state and model_choice == "Random Forest":
                selected_model = st.session_state.rf_model
            elif 'log_model' in st.session_state and model_choice == "Régression Logistique":
                selected_model = st.session_state.log_model
            elif 'mlp_model' in st.session_state and model_choice == "Réseau de Neurones":
                selected_model = st.session_state.mlp_model
            else:
                st.error("Le modèle sélectionné n'est pas disponible.")

            if selected_model:
                accuracy, precision, recall, f1 = evaluate_model(selected_model, st.session_state.X_test, st.session_state.y_test)
                st.write(f"**Performance du modèle ({model_choice})**")
                st.write(f"Accuracy : {accuracy:.2f}")
                st.write(f"Précision : {precision:.2f}")
                st.write(f"Rappel : {recall:.2f}")
                st.write(f"F1-Score : {f1:.2f}")

    # Visualisation des données
    if action == "Visualisation des données":
        if st.button("Afficher la carte de chaleur des corrélations"):
            st.subheader("Carte de chaleur des corrélations des données")
            plt.figure(figsize=(10, 8))
            numerical_df = df.select_dtypes(include=[np.number])
            sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
            st.pyplot(plt)

        if st.button("Afficher la matrice de confusion"):
            if 'rf_model' in st.session_state:
                y_pred = st.session_state.rf_model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Churn", "Churn"], 
                            yticklabels=["Non-Churn", "Churn"])
                plt.xlabel("Prédictions")
                plt.ylabel("Vérités")
                st.pyplot(plt)
            else:
                st.error("Le modèle Random Forest n'est pas entraîné.")

# Personnalisation de l'interface avec un thème Streamlit
st.sidebar.title("Personnalisation")
theme_choice = st.sidebar.radio("Choisissez un thème de couleur", ("Classique", "Sombre", "Clair"))

# Personnalisation du style selon le thème choisi
if theme_choice == "Sombre":
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #333;
            color: white;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
elif theme_choice == "Clair":
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f1f1f1;
            color: black;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
