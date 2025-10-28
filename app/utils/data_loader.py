
import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """Charge les données de fraude bancaire"""
    df = pd.read_csv("data/creditcard_cleaned.csv")

    # Ajouter des colonnes utiles pour les filtres
    df['Hour'] = (df['Time'] // 3600) % 24  # Heure de la transaction
    df['Amount_Category'] = pd.cut(df['Amount'],
                                   bins=[0, 10, 50, 100, 500, 1000, float('inf')],
                                   labels=['<10', '10-50', '50-100', '100-500', '500-1000', '>1000'])
    return df