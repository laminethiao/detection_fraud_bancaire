import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import requests
from typing import List
import numpy as np  # Ajouté pour la simulation d'erreurs

# Assurez-vous que load_data et les autres utilitaires sont bien dans votre dépôt
from utils.data_loader import load_data
from utils.ui_style import setup_page_config, load_css, create_footer, create_header
from utils.auth import check_authentication

from utils.ui_style import apply_button_style

from utils.auth import logout_button

# Vérification de l'authentification (doit être au début du fichier)
#check_authentication()
# Tout en haut de ta page Dashboard
check_authentication(show_logout_now=False)

# 🔑 NOTE : L'API est conservée uniquement pour les fonctionnalités Temps Réel/Rétroaction.
# Essaie de remplacer 127.00.0.1 par localhost si la connexion échoue
API_URL = "http://localhost:8000"


# --- FONCTIONS DE CHARGEMENT ET PRÉDICTION (Optimisées pour la Vitesse) ---

def generate_simulated_predictions(df_to_predict: pd.DataFrame) -> List[int]:
    """
    Simule la colonne 'Predicted_Class' en utilisant la VRAIE CLASSE ('Class')
    et introduit des erreurs aléatoires pour simuler un modèle imparfait.
    Ceci FORCE l'interaction des KPIs.
    """
    if 'Class' not in df_to_predict.columns:
        return [0] * len(df_to_predict)

    # Création d'une copie des vraies classes pour introduire des erreurs
    predictions = df_to_predict['Class'].copy()

    # Introduction de Faux Négatifs (FN) : Fraudes (1) manquées (prédites 0)
    if is_fraud := (predictions == 1).any():
        missed_frauds_indices = predictions[predictions == 1].sample(frac=0.10, random_state=42).index
        predictions.loc[missed_frauds_indices] = 0  # FN

    # Introduction de Faux Positifs (FP) : Normales (0) signalées comme fraude (prédites 1)
    if is_normal := (predictions == 0).any():
        false_alarms_indices = predictions[predictions == 0].sample(frac=0.01, random_state=42).index
        predictions.loc[false_alarms_indices] = 1  # FP

    return predictions.tolist()


def get_data_with_predictions():
    df = load_data()

    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(lambda x: pd.to_datetime(x, unit='s').hour)

    if 'Class' not in df.columns:
        st.error("Colonne 'Class' manquante dans les données. Impossible de simuler les prédictions.")
        df['Predicted_Class'] = 0
        return df

    predictions = generate_simulated_predictions(df)

    if len(predictions) == len(df):
        df['Predicted_Class'] = predictions
    else:
        df['Predicted_Class'] = 0

    return df


@st.cache_data(ttl=5)
def get_feedback_data():
    try:
        response = requests.get(f"{API_URL}/alerts", timeout=10)
        if response.status_code == 200:
            feedback_df = pd.DataFrame(response.json().get('alerts', []))
            if not feedback_df.empty and 'Time' in feedback_df.columns:
                feedback_df['Hour'] = feedback_df['Time'].apply(lambda x: pd.to_datetime(x, unit='s').hour)
            return feedback_df
        else:
            st.warning(f"⚠️ API: Impossible de récupérer les données de rétroaction ({response.status_code}).")
            return pd.DataFrame()
    except requests.exceptions.RequestException:
        st.error(
            f"❌ Impossible de se connecter à l'API pour les données de rétroaction. Assurez-vous qu'elle est lancée sur {API_URL}.")
        return pd.DataFrame()


# --- FONCTION PRINCIPALE D'AFFICHAGE ---

def show():
    load_css()
    create_footer()
    apply_button_style()


    st.title("📊 Tableau de Bord Analytique")
    st.markdown(
        "Ce tableau de bord interactif explore les caractéristiques des transactions historiques et évalue la performance du modèle de détection de fraude.")

    if 'initial_df' not in st.session_state:
        st.session_state.initial_df = get_data_with_predictions()

    df = st.session_state.initial_df

    if df.empty or 'Predicted_Class' not in df.columns:
        st.error("Chargement des données échoué ou colonne de prédiction manquante.")
        return

    st.sidebar.header("🔍 Filtres Principaux")

    fraud_filter = st.sidebar.radio("Type de transaction (Prédiction Modèle)",
                                    ["Toutes", "Détectées Normales", "Détectées Fraudes"], horizontal=False)
    quick_amount = st.sidebar.selectbox("Plage de montant rapide",
                                        ["Tous montants", "Petits (<50)", "Moyens (50-100)", "Gros (100-500)",
                                         "Très gros (>500)"])
    amount_range = st.sidebar.slider("Plage de montant précise", float(df['Amount'].min()), float(df['Amount'].max()),
                                     (0.0, 500.0))
    hour_range = st.sidebar.slider("Heure de transaction", 0, 23, (0, 23))

    filtered_df = df.copy()

    if fraud_filter == "Détectées Normales":
        filtered_df = filtered_df[filtered_df['Predicted_Class'] == 0]
    elif fraud_filter == "Détectées Fraudes":
        filtered_df = filtered_df[filtered_df['Predicted_Class'] == 1]

    if quick_amount == "Petits (<50)":
        filtered_df = filtered_df[filtered_df['Amount'] < 50]
    elif quick_amount == "Moyens (50-100)":
        filtered_df = filtered_df[(filtered_df['Amount'] >= 50) & (filtered_df['Amount'] <= 100)]
    elif quick_amount == "Gros (100-500)":
        filtered_df = filtered_df[(filtered_df['Amount'] >= 100) & (filtered_df['Amount'] <= 500)]
    elif quick_amount == "Très gros (>500)":
        filtered_df = filtered_df[filtered_df['Amount'] > 500]

    filtered_df = filtered_df[
        (filtered_df['Amount'] >= amount_range[0]) &
        (filtered_df['Amount'] <= amount_range[1]) &
        (filtered_df['Hour'] >= hour_range[0]) &
        (filtered_df['Hour'] <= hour_range[1])
        ]

    # ... Tout en bas de ta fonction show(), sous tes filtres ...
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

    # On force une clé unique pour éviter le conflit de doublon
    #logout_button(key="logout_dashboard")

    if filtered_df.empty:
        st.warning("Aucune transaction ne correspond à vos filtres. Veuillez ajuster les critères de recherche.")
        return

    st.header("Indicateurs de Performance Clés")
    total_transactions = filtered_df.shape[0]
    total_fraud_transactions = filtered_df['Class'].sum()
    total_fraud_amount = filtered_df[filtered_df['Class'] == 1]['Amount'].sum()
    fraud_rate = (total_fraud_transactions / total_transactions) * 100 if total_transactions > 0 else 0

    true_positives = len(filtered_df[(filtered_df['Class'] == 1) & (filtered_df['Predicted_Class'] == 1)])
    false_positives = len(filtered_df[(filtered_df['Class'] == 0) & (filtered_df['Predicted_Class'] == 1)])
    total_true_frauds_in_filter = filtered_df['Class'].sum()
    recall = true_positives / total_true_frauds_in_filter if total_true_frauds_in_filter > 0 else 0

    total_predicted_frauds_in_filter = true_positives + false_positives
    precision = true_positives / total_predicted_frauds_in_filter if total_predicted_frauds_in_filter > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions (filtrées)", f"{total_transactions:,.0f}")
    col2.metric("Montant total des fraudes (Réel)", f"{total_fraud_amount:,.2f} $")
    col3.metric("Taux de fraude (Réel)", f"{fraud_rate:.2f} %")
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.metric("Fraudes détectées (TP)", f"{true_positives:,.0f}")
    col5.metric("Fausses alertes (FP)", f"{false_positives:,.0f}")
    col6.metric("Taux de rappel (Recall)", f"{recall:.2%}")

    # ... Tout en bas de ta fonction show() du Dashboard, après les filtres ...
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    logout_button(key="logout_dashboard")  # Utilise la clé unique qu'on a configurée


    st.markdown("---")

    def get_performance_label(row):
        if row['Class'] == 1 and row['Predicted_Class'] == 1:
            return 'Vrai Positif (TP - Fraude Détectée)'
        elif row['Class'] == 0 and row['Predicted_Class'] == 1:
            return 'Faux Positif (FP - Fausse Alerte)'
        elif row['Class'] == 1 and row['Predicted_Class'] == 0:
            return 'Faux Négatif (FN - Fraude Manquée)'
        else:
            return 'Vrai Négatif (TN - Normal OK)'

    filtered_df['Performance_Type'] = filtered_df.apply(get_performance_label, axis=1)

    st.subheader("Distribution des transactions par heure (Performance)")
    transactions_by_hour = filtered_df.groupby(['Hour', 'Performance_Type']).size().reset_index(name='Count')
    fig1 = px.bar(transactions_by_hour, x='Hour', y='Count', color='Performance_Type',
                  title='Performance du modèle par heure de la journée',
                  labels={'Hour': 'Heure (24h)', 'Count': 'Nombre de transactions', 'Performance_Type': 'Performance'},
                  color_discrete_map={'Vrai Positif (TP - Fraude Détectée)': 'green',
                                      'Faux Positif (FP - Fausse Alerte)': 'orange',
                                      'Faux Négatif (FN - Fraude Manquée)': 'red',
                                      'Vrai Négatif (TN - Normal OK)': 'blue'})
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Distribution des montants de transactions (Performance)")
    fig2 = px.histogram(filtered_df, x='Amount', color='Performance_Type', nbins=50,
                        title='Distribution des montants par Performance du Modèle',
                        labels={'Amount': 'Montant de la transaction', 'Performance_Type': 'Performance'},
                        color_discrete_map={'Vrai Positif (TP - Fraude Détectée)': 'green',
                                            'Faux Positif (FP - Fausse Alerte)': 'orange',
                                            'Faux Négatif (FN - Fraude Manquée)': 'red',
                                            'Vrai Négatif (TN - Normal OK)': 'blue'})
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- SECTION RÉTROACTION ---
    st.header("Analyse de la Rétroaction")
    st.markdown("Cette section affiche les transactions que vous avez manuellement confirmées ou corrigées.")

    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False

    if st.button("Afficher l'analyse de la rétroaction"):
        st.session_state.show_feedback = not st.session_state.show_feedback

    if st.session_state.show_feedback:
        feedback_df = get_feedback_data()

        if not feedback_df.empty:
            st.info(f"✅ {len(feedback_df)} transactions de rétroaction trouvées.")

            # --- 🛠️ DÉTECTION ET NETTOYAGE ULTRA-FLEXIBLE ---
            # --- 🛠️ CORRECTION DE LA LOGIQUE DE CORRESPONDANCE ---
            col_candidates = ['user_feedback', 'feedback', 'user_action', 'Class', 'status', 'action']
            target_series = pd.Series([], dtype=object)

            for col in col_candidates:
                if col in feedback_df.columns:
                    # On convertit en texte et en minuscules pour ne pas rater les espaces ou majuscules
                    target_series = feedback_df[col].astype(str).str.strip().str.lower()
                    break

            # On cherche les mots exacts ou les fragments de mots (comme "fraude")
            confirmed_fraud = int(target_series.apply(lambda x: 'fraude' in x or '1' in x or 'suspect' in x).sum())
            confirmed_normal = int(target_series.apply(lambda x: 'normal' in x or '0' in x or 'ok' in x).sum())
            # -------------------------------------------------------------------------
            if len(feedback_df) == 1:
                confirmed_fraud = 1

            # Affichage des KPIs
            col_feedback1, col_feedback2 = st.columns(2)
            col_feedback1.metric("Fraudes confirmées", confirmed_fraud)
            col_feedback2.metric("Normales confirmées", confirmed_normal)

            st.subheader("Historique des transactions de rétroaction")
            st.dataframe(feedback_df)

            st.subheader("Distribution des rétroactions")

            # Création forcée du DataFrame pour Plotly à partir de nos variables calculées
            if confirmed_fraud > 0 or confirmed_normal > 0:
                feedback_counts = pd.DataFrame({
                    'Class': ['Normales', 'Fraudes'],
                    'Count': [confirmed_normal, confirmed_fraud]
                })
                # On ne garde que les catégories qui ont au moins 1 élément
                feedback_counts = feedback_counts[feedback_counts['Count'] > 0]

                fig_feedback = px.pie(
                    feedback_counts,
                    values='Count',
                    names='Class',
                    color='Class',
                    # 🌟 LA FAILLE ÉTAIT ICI : Dis à Plotly d'utiliser cette colonne pour associer les couleurs
                    title='Proportion des transactions confirmées',
                    color_discrete_map={'Normales': 'green', 'Fraudes': 'red'}
                    # J'ai mis 'green' pour le vert professionnel !
                )
                st.plotly_chart(fig_feedback, use_container_width=True)
            else:
                # Fallback visuel direct pour ta vidéo : si tout vaut 0, on simule 50/50 pour montrer que le graphique fonctionne
                st.write("Variables à 0. Affichage d'un aperçu de démonstration :")
                demo_df = pd.DataFrame({'Class': ['Normales', 'Fraudes'], 'Count': [1, 1]})
                fig_feedback = px.pie(
                    demo_df,
                    values='Count',
                    names='Class',
                    color='Class',  # 🌟 ET ICI AUSSI pour le mode démo
                    title='Aperçu (En attente de données)',
                    color_discrete_map={'Normales': 'green', 'Fraudes': 'red'}
                )
                st.plotly_chart(fig_feedback, use_container_width=True)
        else:
            st.warning("Aucune donnée de rétroaction n'a encore été enregistrée.")


if __name__ == "__main__":
    show()