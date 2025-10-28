import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import requests
from utils.data_loader import load_data
from utils.ui_style import setup_page_config, load_css, create_footer, create_header

from utils.auth import check_authentication

check_authentication()

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000"

@st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, "..")
    model_path = os.path.join(parent_dir, "models", "xgb_fraud_detection_model.pkl")
    scaler_path = os.path.join(parent_dir, "models", "scaler.pkl")

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Erreur : Un des fichiers du modèle n'a pas été trouvé. Veuillez vérifier le chemin. {e}")
        return None, None

@st.cache_data
def get_data():
    df = load_data()
    df['Hour'] = df['Time'].apply(lambda x: pd.to_datetime(x, unit='s').hour)
    return df

@st.cache_data(ttl=5)
def get_feedback_data():
    """
    Récupère les données de rétroaction depuis l'API.
    """
    try:
        response = requests.get(f"{API_URL}/alerts")
        if response.status_code == 200:
            return pd.DataFrame(response.json()['alerts'])
        else:
            st.error("Erreur lors de la récupération des données de rétroaction.")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de se connecter à l'API : {e}. Assurez-vous que l'API est en cours d'exécution.")
        return pd.DataFrame()

def show():

    load_css()
    create_footer()

    st.title("📊 Tableau de Bord Analytique")
    st.markdown(
        "Ce tableau de bord interactif vous permet d'explorer les caractéristiques des transactions et d'évaluer la performance du modèle de détection de fraude.")

    # Afficher la vue principale du tableau de bord
    df = get_data()

    st.sidebar.header("🔍 Filtres Principaux")
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

    # --- PRÉDICTION SUR LES DONNÉES FILTRÉES ---
    model, scaler = load_resources()

    if model is None or scaler is None:
        st.error("Les ressources du modèle n'ont pas pu être chargées. Contactez l'administrateur.")
    elif filtered_df.empty:
        st.warning("Aucune transaction ne correspond à vos filtres. Veuillez ajuster les critères de recherche.")
        filtered_df['Predicted_Class'] = 0
    else:
        try:
            # Créer une copie du DataFrame pour ne pas modifier l'original
            features = filtered_df.copy()

            # Normaliser les variables 'Time' et 'Amount' ENSEMBLE
            features_to_scale = ['Time', 'Amount']
            features[features_to_scale] = scaler.transform(features[features_to_scale])

            # Créer la fonctionnalité 'Amount_Category' si le modèle l'attend
            if 'Amount_Category' in model.feature_names_in_:
                bins = [0, 50, 100, 500, float('inf')]
                labels = [0, 1, 2, 3]
                features['Amount_Category'] = pd.cut(features['Amount'], bins=bins, labels=labels, right=False).astype(int)

            # S'assurer que les colonnes sont dans le bon ordre avant de faire la prédiction
            final_features = features[list(model.feature_names_in_)]

            # Faire la prédiction
            predictions = model.predict(final_features)
            filtered_df['Predicted_Class'] = predictions
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de la prédiction du modèle. Erreur: {e}")
            filtered_df['Predicted_Class'] = 0

    # --- AFFICHAGE DES KPIS ET VISUALISATIONS ---
    st.header("Indicateurs de Performance Clés")

    total_transactions = filtered_df.shape[0]
    total_fraud_transactions = filtered_df['Class'].sum()
    total_fraud_amount = filtered_df[filtered_df['Class'] == 1]['Amount'].sum()
    fraud_rate = (total_fraud_transactions / total_transactions) * 100 if total_transactions > 0 else 0
    true_positives = len(filtered_df[(filtered_df['Class'] == 1) & (filtered_df['Predicted_Class'] == 1)])
    false_positives = len(filtered_df[(filtered_df['Class'] == 0) & (filtered_df['Predicted_Class'] == 1)])
    recall = true_positives / total_fraud_transactions if total_fraud_transactions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions (filtrées)", f"{total_transactions:,.0f}")
    col2.metric("Montant total des fraudes", f"{total_fraud_amount:,.2f} $")
    col3.metric("Taux de fraude", f"{fraud_rate:.2f} %")
    col4, col5, col6 = st.columns(3)
    col4.metric("Fraudes détectées", f"{true_positives:,.0f}")
    col5.metric("Fausses alertes", f"{false_positives:,.0f}")
    col6.metric("Taux de rappel (Recall)", f"{recall:.2%}")

    st.markdown("---")
    st.header("Visualisations Clés")

    if total_transactions > 0:
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
            st.markdown("#### Télécharger les Données Filtrées")
            st.info("Aperçu des 10 premières lignes. Le fichier CSV complet contient toutes les transactions filtrées.")
            st.dataframe(filtered_df.head(10), use_container_width=True)
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
