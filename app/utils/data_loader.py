import pandas as pd
import streamlit as st
import os


@st.cache_data
def load_data():
    """Charge les données de fraude bancaire en utilisant un chemin absolu pour le déploiement Cloud."""

    # Construction du chemin absolu à partir de la position du fichier data_loader.py

    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'creditcard_cleaned.csv')

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        # Message d'erreur clair si le fichier n'est pas trouvé
        st.error(f"Erreur: Fichier de données introuvable à {data_path}. Veuillez vérifier le chemin sur le dépôt.")
        return pd.DataFrame()  # Retourne un DataFrame vide pour éviter le crash de l'application

    # Ajouter des colonnes utiles pour les filtres
    df['Hour'] = (df['Time'] // 3600) % 24  # Heure de la transaction
    df['Amount_Category'] = pd.cut(df['Amount'],
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                   labels=['<10', '10-50', '50-100', '100-500', '500-1000', '>1000'])
    return df
