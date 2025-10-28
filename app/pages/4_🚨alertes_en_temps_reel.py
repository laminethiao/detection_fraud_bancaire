
import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np

# Import des fonctions de style à partir d'un autre fichier
from utils.ui_style import setup_page_config, load_css, create_footer, apply_button_style

from utils.auth import check_authentication



check_authentication()

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000"


@st.cache_data(ttl=5)
def get_model_alerts():
    """
    Récupère la liste des alertes de fraude à partir de l'API.
    La fonction est mise en cache pour 5 secondes pour un rafraîchissement régulier.
    """
    try:
        response = requests.get(f"{API_URL}/alerts")
        if response.status_code == 200:
            alerts_data = response.json()['alerts']
            if not alerts_data:
                return pd.DataFrame()
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df['id'] = alerts_df.index.astype(str)
            return alerts_df
        else:
            st.error(f"Erreur lors de la récupération des alertes : {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de se connecter à l'API. Assurez-vous que l'API est en cours d'exécution.")
        return pd.DataFrame()


@st.cache_data
def get_historical_data():
    """
    Récupère des données historiques pour la visualisation. Tente d'abord de se connecter
    à l'API, puis charge le fichier local en cas d'échec.
    """
    file_path = "data/creditcard_cleaned.csv"

    with st.spinner("Chargement des données historiques..."):
        try:
            response = requests.get(f"{API_URL}/historical_data", timeout=5)
            if response.status_code == 200:
                return pd.DataFrame(response.json()['data'])
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            pass

    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error(f"Erreur : Impossible de trouver le fichier '{file_path}' localement.")
        return pd.DataFrame()


def find_most_anomalous_feature(current_transaction, historical_df):
    """
    Trouve la caractéristique (V1-V28) qui est le plus en dehors de la distribution normale.
    Retourne la caractéristique et son Z-score.
    """
    if historical_df.empty:
        return 'V1', 0.0

    anomalies = {}
    normal_data = historical_df[historical_df['Class'] == 0]

    for feature in [f"V{i}" for i in range(1, 29)]:
        if feature in normal_data.columns and feature in current_transaction:
            mean = normal_data[feature].mean()
            std = normal_data[feature].std()

            if std > 0:
                z_score = abs(current_transaction[feature] - mean) / std
                anomalies[feature] = z_score

    if anomalies:
        most_anomalous_feature = max(anomalies, key=anomalies.get)
        return most_anomalous_feature, anomalies[most_anomalous_feature]
    else:
        return 'V1', 0.0


def create_pca_plot(df, current_transaction, feature):
    """Crée un graphique de distribution pour une valeur PCA."""
    fig = go.Figure()

    if df.empty:
        st.error("Impossible de créer le graphique car les données historiques sont manquantes.")
        return

    normal_data = df[df['Class'] == 0]
    fraud_data = df[df['Class'] == 1]

    # Générer la courbe de densité pour les transactions normales
    if not normal_data.empty:
        kde = gaussian_kde(normal_data[feature])
        x_vals = np.linspace(normal_data[feature].min(), normal_data[feature].max(), 1000)
        y_vals = kde.evaluate(x_vals)

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines',
            name='Distribution Normale',
            fill='tozeroy',
            line_color='#28a745',
            opacity=0.6
        ))

    # Ajouter l'histogramme des transactions frauduleuses (pour référence)
    fig.add_trace(go.Histogram(
        x=fraud_data[feature],
        name='Transactions Frauduleuses',
        marker_color='#dc3545',
        opacity=0.6,
        histnorm='probability density'
    ))

    # Ajouter la ligne de la transaction actuelle
    fig.add_vline(
        x=current_transaction[feature],
        line_dash="dash",
        line_color="black",
        annotation_text=f"Transaction actuelle: {current_transaction[feature]:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title_text=f"Distribution de la caractéristique '{feature}'",
        xaxis_title_text=feature,
        yaxis_title_text='Densité',
        barmode='overlay',
        legend_title_text="Légende",
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    st.plotly_chart(fig, use_container_width=True)


def send_feedback(transaction_id, transaction_data, true_class, message):
    """
    Envoie la rétroaction à l'API et affiche un message de confirmation personnalisé.
    """
    try:
        feedback_data = transaction_data.to_dict()
        feedback_data['Class'] = true_class

        response = requests.post(f"{API_URL}/feedback", json=feedback_data)
        if response.status_code == 200:
            st.success(message)
            time.sleep(2)
            if 'alerts_queue' in st.session_state and st.session_state.alerts_queue:
                st.session_state.alerts_queue.pop(0)
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"Échec de l'enregistrement de la rétroaction : {response.status_code}")
            st.error(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'envoi de la rétroaction à l'API : {e}")


# ... (Imports et fonctions utilitaires comme get_model_alerts, get_historical_data, send_feedback, etc., sont inchangés) ...

def show():
    """Affiche la page des alertes en temps réel avec des améliorations interactives."""
    # setup_page_config()
    load_css()
    apply_button_style()
    create_footer()

    st.title("🚨 Centre de Triage des Alertes")
    st.markdown(
        "Bienvenue dans votre file d'attente d'alertes. Validez les transactions suspectes une par une pour les retirer de la liste.")
    st.markdown("---")

    # --- Gestion de la file d'attente ---
    if 'alerts_queue' not in st.session_state:
        alerts_df = get_model_alerts()
        st.session_state.alerts_queue = alerts_df.to_dict('records')
        st.session_state.initial_alerts_count = len(st.session_state.alerts_queue)

    historical_df = get_historical_data()
    alerts_queue = st.session_state.alerts_queue

    if not alerts_queue:
        st.info(
            "Félicitations, toutes les alertes ont été traitées ! Allez sur la page 'Détection' pour en créer de nouvelles.")
        st.session_state.initial_alerts_count = 0
    else:
        remaining_alerts = len(alerts_queue)
        initial_alerts_count = st.session_state.initial_alerts_count

        progress_value = 1.0 - (remaining_alerts / initial_alerts_count) if initial_alerts_count > 0 else 1.0
        st.progress(progress_value, text=f"**{remaining_alerts} alerte(s)** restante(s) à traiter")

        current_transaction_data = alerts_queue[0]
        current_transaction = pd.Series(current_transaction_data)

        # Déterminer la caractéristique la plus anormale (pour le défaut et l'explication)
        most_anomalous_feature, z_score = find_most_anomalous_feature(current_transaction, historical_df)
        model_verdict = 1 if z_score > 3 else 0  # Maintien de la logique de verdict

        # --- Affichage des informations clés ---
        st.markdown('<div class="card">', unsafe_allow_html=True)  # Utilisation de la classe .card de ui_style.py

        col_info_1, col_info_2, col_info_3 = st.columns([1.5, 1, 3])

        with col_info_1:
            st.metric("ID de la transaction", current_transaction['id'])
        with col_info_2:
            st.metric("Montant", f"{current_transaction['Amount']:.2f} $")

        with col_info_3:
            # Afficher l'explication du modèle
            if model_verdict == 1:
                st.markdown(f"""
                    <div class='fraud-alert'>
                        <b>Verdict Modèle : SUSPECTÉ DE FRAUDE</b><br>
                        Raison : {most_anomalous_feature} (Z-score: {z_score:.2f}) est hors norme.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='no-fraud'>
                        <b>Verdict Modèle : NORMAL</b><br>
                        Raison : {most_anomalous_feature} (Z-score: {z_score:.2f}) est dans la norme.
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")  # Séparateur

        # --- Visualisation Interactive ---
        st.subheader("Visualisation de l'Anomalie (Analyse de la Distribution)")

        if not historical_df.empty:
            # 1. Sélecteur de Caractéristique
            all_v_features = [f"V{i}" for i in range(1, 29)]

            # Utiliser la caractéristique la plus anormale comme valeur par défaut
            selected_feature = st.selectbox(
                "Choisir la caractéristique PCA à analyser :",
                options=all_v_features,
                index=all_v_features.index(most_anomalous_feature),
                key=f"feature_selector_{current_transaction['id']}"
            )

            # 2. Affichage du Graphique
            create_pca_plot(historical_df, current_transaction, feature=selected_feature)

        # --- Affichage des Valeurs PCA sous forme de Tableau ---
        st.markdown("---")
        with st.expander("Voir toutes les valeurs PCA (pour un examen détaillé)"):

            v_data = {f"V{i}": current_transaction[f"V{i}"] for i in range(1, 29)}
            v_df = pd.DataFrame(v_data.items(), columns=['Caractéristique', 'Valeur'])

            # Afficher les valeurs V en utilisant le style Streamlit pour plus de propreté
            st.dataframe(v_df.T, use_container_width=True)  # Transposée pour un meilleur affichage

        st.markdown("---")

        # --- Boutons de Rétroaction ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚨 Confirmer FRAUDE (Class=1)", key=f"fraud_{current_transaction['id']}",
                         help="Cliquez pour valider la fraude. La transaction est retirée de la file.", type='primary'):
                with st.spinner("Envoi de la rétroaction..."):
                    send_feedback(current_transaction['id'], current_transaction.drop('id'), 1,
                                  "Rétroaction de *fraude* enregistrée avec succès ! Redirection...")
        with col2:
            if st.button("✅ Confirmer NORMAL (Class=0)", key=f"normal_{current_transaction['id']}",
                         help="Cliquez pour confirmer que la transaction est normale. La transaction est retirée de la file.",
                         type='secondary'):
                with st.spinner("Envoi de la rétroaction..."):
                    send_feedback(current_transaction['id'], current_transaction.drop('id'), 0,
                                  "Rétroaction de transaction *normale* enregistrée avec succès ! Redirection...")

        st.markdown('</div>', unsafe_allow_html=True)




if __name__ == "__main__":
    show()
