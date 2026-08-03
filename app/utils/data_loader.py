import os
import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """Charge les données de fraude bancaire (Version robuste local & déploiement)."""

    # 1. Détection dynamique du chemin absolu de ce fichier (data_loader.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Test des deux structures de dossiers les plus probables
    # Option 1 : si data_loader.py est dans un sous-dossier (ex: app/utils/)
    path_option1 = os.path.join(current_dir, "..", "..", "data", "creditcard_cleaned.csv")
    # Option 2 : si data_loader.py est plus proche de la racine (ex: utils/)
    path_option2 = os.path.join(current_dir, "..", "data", "creditcard_cleaned.csv")

    # Sélection du bon chemin selon l'existence réelle du fichier
    if os.path.exists(path_option1):
        data_path = path_option1
    elif os.path.exists(path_option2):
        data_path = path_option2
    else:
        # Repli par défaut si la structure est différente
        data_path = "data/creditcard_cleaned.csv"

    # 3. Chargement du fichier CSV
    df = pd.read_csv(data_path)

    # 4. Ajouter des colonnes utiles pour les filtres (Ton code d'origine)
    df['Hour'] = (df['Time'] // 3600) % 24  # Heure de la transaction
    df['Amount_Category'] = pd.cut(df['Amount'],
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                   labels=['<10', '10-50', '50-100', '100-500', '500-1000', '>1000'])
    return df