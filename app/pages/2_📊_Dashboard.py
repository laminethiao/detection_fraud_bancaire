import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import requests
from utils.data_loader import load_data
from utils.ui_style import setup_page_config, load_css, create_footer, create_header
from utils.auth import check_authentication

check_authentication()

# URL de l'API FastAPI
API_URL = "http://detection-fraud-bancaire.fly.dev"

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

def predict_transaction_api(transaction_data):
    """
    Prédit si une transaction est frauduleuse via l'API.
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction_data)
        if response.status_code == 200:
            return response.json()['prediction']
        else:
            st.error(f"Erreur API: {response.status_code}")
            return 0
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return 0

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

    # --- PRÉDICTION SUR LES DONNÉES FILTRÉES VIA L'API ---
    if filtered_df.empty:
        st.warning("Aucune transaction ne correspond à vos filtres. Veuillez ajuster les critères de recherche.")
        filtered_df['Predicted_Class'] = 0
    else:
        try:
            # Utiliser l'API pour les prédictions
            predictions = []
            
            # Barre de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = len(filtered_df)
            for i, (_, row) in enumerate(filtered_df.iterrows()):
                # Mettre à jour la progression
                progress = (i + 1) / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Prédiction en cours... {i+1}/{total_rows}")
                
                # Préparer les données pour l'API
                transaction_data = {
                    "Time": float(row['Time']),
                    "V1": float(row['V1']), "V2": float(row['V2']), "V3": float(row['V3']), "V4": float(row['V4']),
                    "V5": float(row['V5']), "V6": float(row['V6']), "V7": float(row['V7']), "V8": float(row['V8']),
                    "V9": float(row['V9']), "V10": float(row['V10']), "V11": float(row['V11']), "V12": float(row['V12']),
                    "V13": float(row['V13']), "V14": float(row['V14']), "V15": float(row['V15']), "V16": float(row['V16']),
                    "V17": float(row['V17']), "V18": float(row['V18']), "V19": float(row['V19']), "V20": float(row['V20']),
                    "V21": float(row['V21']), "V22": float(row['V22']), "V23": float(row['V23']), "V24": float(row['V24']),
                    "V25": float(row['V25']), "V26": float(row['V26']), "V27": float(row['V27']), "V28": float(row['V28']),
                    "Amount": float(row['Amount'])
                }
                
                # Appel à l'API
                prediction = predict_transaction_api(transaction_data)
                predictions.append(prediction)
            
            # Nettoyer la barre de progression
            progress_bar.empty()
            status_text.empty()
            
            filtered_df['Predicted_Class'] = predictions
            st.success("✅ Prédictions terminées avec succès !")
            
        except Exception as e:
            st.error(f"Erreur lors de l'appel à l'API : {e}")
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
            st.markdown("◆ Télécharger les Données Filtrées")
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