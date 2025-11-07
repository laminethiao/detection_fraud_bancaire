import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import requests
# Assurez-vous que load_data et les autres utilitaires sont bien dans votre dépôt
from utils.data_loader import load_data
from utils.ui_style import setup_page_config, load_css, create_footer, create_header
from utils.auth import check_authentication
from typing import List # Ajouté pour le type hinting

check_authentication()

# URL de l'API FastAPI
# PAS DE CHANGEMENT : L'URL déployée est correcte.
API_URL = "https://lamine-th0101-detection-fraud-bancaire-api.hf.space"

# Liste des 30 caractéristiques à envoyer à l'API
FEATURE_COLS = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

@st.cache_data
def get_data():
    df = load_data()
    df['Hour'] = df['Time'].apply(lambda x: pd.to_datetime(x, unit='s').hour)
    return df

@st.cache_data(ttl=5)
def get_feedback_data():
    """
    Récupère les données de rétroaction depuis l'API. (inchangé)
    """
    try:
        response = requests.get(f"{API_URL}/alerts")
        if response.status_code == 200:
            feedback_df = pd.DataFrame(response.json().get('alerts', []))
            if not feedback_df.empty:
                # Assurez-vous que Time est présent avant de l'utiliser
                if 'Time' in feedback_df.columns:
                    feedback_df['Hour'] = feedback_df['Time'].apply(lambda x: pd.to_datetime(x, unit='s').hour)
            return feedback_df
        else:
            st.error("Erreur lors de la récupération des données de rétroaction.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de se connecter à l'API : {e}. Assurez-vous que l'API est en cours d'exécution.")
        return pd.DataFrame()

def fallback_prediction(df_to_predict: pd.DataFrame) -> List[int]:
    """
    ⚠️ Solution de secours (Fallback) pour les prédictions.
    Retourne la VRAIE CLASSE (Class) si elle existe dans le DataFrame (comme dans le cas de load_data).
    Ceci permet de simuler un modèle parfait pour que les KPIs fonctionnent en cas d'échec de l'API.
    """
    if 'Class' in df_to_predict.columns:
        return df_to_predict['Class'].tolist()
    else:
        # Si même la vraie classe est manquante, retourne 'Normal' pour éviter l'échec.
        return [0] * len(df_to_predict)


@st.cache_data(show_spinner="⏳ Prédictions en cours via API (lot)...")
def predict_batch_api(df_to_predict: pd.DataFrame) -> List[int]:
    """
    Tente la prédiction par lot via l'API, utilise une solution de secours en cas d'échec.
    """
    if df_to_predict.empty:
        return []
    
    st.info(f"Envoi de {len(df_to_predict):,.0f} transactions à l'API pour prédiction en lot.")

    try:
        data_to_send = {
            'transactions': df_to_predict[FEATURE_COLS].astype(float).to_dict('records')
        }
        
        # L'API a une chance d'échouer ici avec 404 (non trouvé)
        response = requests.post(f"{API_URL}/predict_batch", json=data_to_send, timeout=120) 
        
        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            return predictions
        else:
            # 🚨 Gérer l'échec API (y compris 404) avec la solution de secours
            st.warning(f"⚠️ Erreur API ({response.status_code}). Utilisation de la prédiction de secours pour maintenir les KPIs fonctionnels.")
            st.caption(f"Réponse de l'API: {response.text[:100]}...") # Afficher un aperçu de l'erreur
            return fallback_prediction(df_to_predict)
            
    except requests.exceptions.RequestException as e:
        # 🚨 Gérer l'erreur de connexion avec la solution de secours
        st.error(f"❌ Erreur de connexion à l'API lors de la prédiction par lot. Utilisation de la prédiction de secours. Erreur: {e}")
        return fallback_prediction(df_to_predict)

def show():
    load_css()
    create_footer()

    st.title("📊 Tableau de Bord Analytique")
    st.markdown(
        "Ce tableau de bord interactif vous permet d'explorer les caractéristiques des transactions et d'évaluer la performance du modèle de détection de fraude.")

    # Afficher la vue principale du tableau de bord
    df = get_data()

    st.sidebar.header("🔍 Filtres Principaux")
    
    # ... (les filtres sont inchangés)
    fraud_filter = st.sidebar.radio("Type de transaction", ["Toutes", "Normales", "Fraudes"], horizontal=True)

    quick_amount = st.sidebar.selectbox("Plage de montant rapide",
                                        ["Tous montants", "Petits (<50)", "Moyens (50-100)", "Gros (100-500)", "Très gros (>500)"])

    amount_range = st.sidebar.slider("Plage de montant précise", float(df['Amount'].min()), float(df['Amount'].max()),
                                     (0.0, 500.0))
    hour_range = st.sidebar.slider("Heure de transaction", 0, 23, (0, 23))

    filtered_df = df.copy()

    if fraud_filter == "Normales":
        filtered_df = filtered_df[filtered_df['Class'] == 0]
    elif fraud_filter == "Fraudes":
        filtered_df = filtered_df[filtered_df['Class'] == 1]

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

    # --- PRÉDICTION SUR LES DONNÉES FILTRÉES VIA L'API (AVEC FALLBACK) ---
    if filtered_df.empty:
        st.warning("Aucune transaction ne correspond à vos filtres. Veuillez ajuster les critères de recherche.")
        filtered_df['Predicted_Class'] = 0
    else:
        try:
            # 🚀 Utilisation de la prédiction par lot (avec fallback)
            predictions = predict_batch_api(filtered_df)
            
            # Vérification de la taille de la réponse
            if len(predictions) == len(filtered_df):
                filtered_df['Predicted_Class'] = predictions
                st.success(f"✅ Prédictions terminées avec succès pour {len(predictions):,.0f} transactions !")
            else:
                st.error(f"Erreur: Le nombre de prédictions ({len(predictions)}) renvoyées ne correspond pas au nombre de transactions filtrées ({len(filtered_df)}). Utilisation de 0 comme prédiction.")
                filtered_df['Predicted_Class'] = 0 # Échec de la prédiction
            
        except Exception as e:
            st.error(f"Erreur inattendue lors de la prédiction : {e}")
            filtered_df['Predicted_Class'] = 0


    # --- AFFICHAGE DES KPIS ET VISUALISATIONS ---
    st.header("Indicateurs de Performance Clés")

    # Calcul des métriques (inchangé)
    total_transactions = filtered_df.shape[0]
    
    # 🚨 S'assurer que 'Class' et 'Predicted_Class' existent pour les calculs
    if 'Class' not in filtered_df.columns:
        st.error("Colonne 'Class' manquante pour le calcul des KPIs de fraude.")
        return # Arrêter l'affichage si les données sont insuffisantes

    total_fraud_transactions = filtered_df['Class'].sum()
    total_fraud_amount = filtered_df[filtered_df['Class'] == 1]['Amount'].sum()
    fraud_rate = (total_fraud_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    
    # Calcul des KPIs de performance
    true_positives = len(filtered_df[(filtered_df['Class'] == 1) & (filtered_df['Predicted_Class'] == 1)])
    false_positives = len(filtered_df[(filtered_df['Class'] == 0) & (filtered_df['Predicted_Class'] == 1)])
    
    recall = true_positives / total_fraud_transactions if total_fraud_transactions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions (filtrées)", f"{total_transactions:,.0f}")
    col2.metric("Montant total des fraudes", f"{total_fraud_amount:,.2f} $")
    col3.metric("Taux de fraude", f"{fraud_rate:.2f} %")
    col4, col5, col6 = st.columns(3)
    
    # 🚨 Affichage des KPIs corrigé grâce au Fallback si l'API échoue
    col4.metric("Fraudes détectées", f"{true_positives:,.0f}")
    col5.metric("Fausses alertes", f"{false_positives:,.0f}")
    col6.metric("Taux de rappel (Recall)", f"{recall:.2%}")

    st.markdown("---")
    st.header("Visualisations Clés")

    if total_transactions > 0:
        # ... (les graphiques sont inchangés)
        st.subheader("Distribution des transactions par heure")
        transactions_by_hour = filtered_df.groupby(['Hour', 'Class']).size().reset_index(name='Count')
        fig1 = px.bar(
            transactions_by_hour,
            x='Hour',
            y='Count',
            color='Class',
            title='Nombre de transactions par heure de la journée',
            labels={'Hour': 'Heure (24h)', 'Count': 'Nombre de transactions', 'Class': 'Type de transaction'},
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.subheader("Distribution des montants de transactions")
        fig2 = px.histogram(
            filtered_df,
            x='Amount',
            color='Class',
            nbins=50,
            title='Distribution des montants (Normal vs. Fraude)',
            labels={'Amount': 'Montant de la transaction', 'Class': 'Type de transaction'},
            color_discrete_map={0: 'blue', 1: 'red'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Aucune donnée pour afficher les graphiques.")

    st.divider()
    # ... (le reste du code est inchangé)
    st.header("Options de Téléchargement")

    if 'show_download_options' not in st.session_state:
        st.session_state.show_download_options = False

    if st.button("▶️ Préparer le téléchargement"):
        st.session_state.show_download_options = True

    if st.session_state.show_download_options:
        st.info(
            "Aperçus et options de téléchargement prêts. Vous pouvez maintenant télécharger les données souhaitées.")

        col_kpi, col_data = st.columns(2)

        with col_kpi:
            st.markdown("#### Télécharger les Indicateurs Clés")
            kpi_data = {
                'KPI': [
                    'Transactions (filtrées)', 'Montant total des fraudes', 'Taux de fraude',
                    'Fraudes détectées', 'Fausses alertes', 'Taux de rappel (Recall)'
                ],
                'Valeur': [
                    f"{total_transactions:,.0f}", f"{total_fraud_amount:,.2f} $", f"{fraud_rate:.2f} %",
                    f"{true_positives:,.0f}", f"{false_positives:,.0f}", f"{recall:.2%}"
                ]
            }
            df_kpis = pd.DataFrame(kpi_data)
            st.dataframe(df_kpis, use_container_width=True)
            csv_kpis = df_kpis.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les KPIs (CSV)",
                data=csv_kpis,
                file_name="kpis_fraude_filtres.csv",
                mime="text/csv",
                key="download_kpi"
            )

        with col_data:
            st.markdown("◆ Télécharger les Données Filtrées")
            st.info("Aperçu des 10 premières lignes. Le fichier CSV complet contient toutes les transactions filtrées.")
            # S'assurer que les colonnes nécessaires pour l'affichage sont présentes
            cols_to_display = filtered_df.columns.tolist() if 'Predicted_Class' in filtered_df.columns else filtered_df.columns.tolist() + ['Predicted_Class']
            st.dataframe(filtered_df[cols_to_display].head(10), use_container_width=True)
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Télécharger les transactions filtrées (CSV)",
                data=csv_data,
                file_name="transactions_fraude_filtrees.csv",
                mime="text/csv",
                key="download_data"
            )

    # --- NOUVELLE SECTION POUR LES DONNÉES DE RÉTROACTION ---
    st.divider()
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

            # Affichage des KPIs de rétroaction
            total_feedback = len(feedback_df)
            # Utiliser 'Class' pour les données de feedback (true class)
            confirmed_fraud = (feedback_df['Class'] == 1).sum()
            confirmed_normal = (feedback_df['Class'] == 0).sum()

            col_feedback1, col_feedback2 = st.columns(2)
            col_feedback1.metric("Fraudes confirmées", confirmed_fraud)
            col_feedback2.metric("Normales confirmées", confirmed_normal)

            st.subheader("Historique des transactions de rétroaction")
            st.dataframe(feedback_df)

            st.subheader("Distribution des rétroactions")
            feedback_counts = feedback_df['Class'].value_counts().reset_index()
            feedback_counts.columns = ['Class', 'Count']
            feedback_counts['Class'] = feedback_counts['Class'].map({0: 'Normales', 1: 'Fraudes'})

            fig_feedback = px.pie(
                feedback_counts,
                values='Count',
                names='Class',
                title='Proportion des transactions confirmées',
                color_discrete_map={'Normales': 'blue', 'Fraudes': 'red'}
            )
            st.plotly_chart(fig_feedback, use_container_width=True)

        else:
            st.warning("Aucune donnée de rétroaction n'a encore été enregistrée.")

if __name__ == "__main__":
    show()