import streamlit as st
import pandas as pd
import requests
import time  # Import conservé mais non utilisé dans la fonction corrigée
import os
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np
from typing import List, Dict, Any  # Ajouté pour la clarté

# Import des fonctions de style à partir d'un autre fichier
# Assurez-vous que ces fichiers existent dans votre structure (utils/...)
from utils.ui_style import setup_page_config, load_css, create_footer, apply_button_style
from utils.auth import check_authentication
from utils.data_loader import load_data

from utils.auth import logout_button

# Assurez-vous d'appeler check_authentication avant tout affichage
# Note : check_authentication() doit être défini et importé correctement
#check_authentication()
# Tout en haut du fichier de la page
check_authentication(show_logout_now=False)

# 🔑 MODIFICATION CRUCIALE : URL de l'API déployée
# Pour le test local (si votre API locale est en cours d'exécution) :
API_URL = "http://localhost:8000"

ALERT_URL = f"{API_URL}/alert"
GET_ALERTS_URL = f"{API_URL}/alerts"
HISTORICAL_DATA_URL = f"{API_URL}/historical_data"


@st.cache_data(ttl=5)
def get_model_alerts() -> pd.DataFrame:
    """Récupère la liste des alertes de fraude à partir de l'API."""
    try:
        # ✅ CORRECTION TIMEOUT : Augmenté de 10 à 30 secondes
        response = requests.get(GET_ALERTS_URL, timeout=30)
        if response.status_code == 200:
            alerts_data = response.json().get('alerts', [])
            if not alerts_data:
                return pd.DataFrame()
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df['id'] = alerts_df.index.astype(str)

            if 'model_prediction' not in alerts_df.columns:
                alerts_df['model_prediction'] = 1

            return alerts_df
        else:
            st.error(f"Erreur lors de la récupération des alertes : {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de se connecter à l'API ({GET_ALERTS_URL}). Vérifiez l'état de l'API. Erreur: {e}")
        return pd.DataFrame()


@st.cache_data
def get_historical_data() -> pd.DataFrame:
    """
    Récupère des données historiques pour la visualisation en direct (version locale rapide).
    """
    with st.spinner("Chargement des données historiques locales..."):
        try:
            # Charge le CSV directement sur le disque (Prend 1 à 2 secondes max)
            df = load_data()
            return df
        except Exception as e:
            st.error(f"Erreur lors du chargement des données locales : {e}")
            return pd.DataFrame()
def find_most_anomalous_feature(current_transaction: pd.Series, historical_df: pd.DataFrame) -> tuple[str, float]:
    """Trouve la caractéristique PCA (V1-V28) qui est la plus éloignée de la moyenne normale."""
    if historical_df.empty or 'Class' not in historical_df.columns:
        return 'V1', 0.0

    anomalies = {}
    normal_data = historical_df[historical_df['Class'] == 0]

    for feature in [f"V{i}" for i in range(1, 29)]:
        if feature in normal_data.columns and feature in current_transaction:
            mean = normal_data[feature].mean()
            std = normal_data[feature].std()

            if std > 0:
                try:
                    current_value = float(current_transaction[feature])
                except ValueError:
                    continue

                # Calcul du Z-score
                z_score = abs(current_value - mean) / std
                anomalies[feature] = z_score

    # --- CODE AVEC LA FAUTE DE FRAPPE ---
    if anomalies:
        most_anomalous_feature = max(anomalies, key=anomalies.get)
        return most_anomalous_feature, anomalies[most_anomalous_feature]
    else:
        return 'V1', 0.0


def create_pca_plot(df: pd.DataFrame, current_transaction: pd.Series, feature: str):
    """Crée un graphique de distribution pour la caractéristique PCA sélectionnée."""
    fig = go.Figure()

    if df.empty or 'Class' not in df.columns or feature not in df.columns:
        st.error("Impossible de créer le graphique car les données historiques sont manquantes ou incomplètes.")
        return

    normal_data = df[df['Class'] == 0]
    fraud_data = df[df['Class'] == 1]

    # 1. Distribution Normale (KDE)
    if not normal_data.empty and len(normal_data[feature].unique()) > 1:
        try:
            kde = gaussian_kde(normal_data[feature].dropna())
            # Déterminer la plage pour la distribution
            x_min = normal_data[feature].min() if normal_data[feature].min() < fraud_data[feature].min() else \
            fraud_data[feature].min()
            x_max = normal_data[feature].max() if normal_data[feature].max() > fraud_data[feature].max() else \
            fraud_data[feature].max()
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = kde.evaluate(x_vals)

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                name='Distribution Normale (KDE)',
                fill='tozeroy',
                line_color='#28a745',
                opacity=0.6
            ))
        except ValueError:
            pass

    # 2. Distribution Frauduleuse (Histogramme)
    if not fraud_data.empty:
        fig.add_trace(go.Histogram(
            x=fraud_data[feature],
            name='Transactions Frauduleuses',
            marker_color='#dc3545',
            opacity=0.6,
            histnorm='probability density'
        ))

    # 3. Ligne de la transaction actuelle
    try:
        current_value = float(current_transaction.get(feature, 0.0))
        fig.add_vline(
            x=current_value,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Transaction actuelle: {current_value:.2f}",
            annotation_position="top right"
        )
    except (ValueError, KeyError):
        pass

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


def submit_feedback(feedback_data: Dict[str, Any]) -> bool:
    """Soumet une rétroaction à l'API en utilisant le nouvel endpoint /alert."""
    try:
        # ✅ CORRECTION TIMEOUT : Augmenté de 10 à 30 secondes
        response = requests.post(ALERT_URL, json=feedback_data, timeout=30)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Erreur lors de l'envoi de la rétroaction: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API lors de l'envoi de feedback: {e}")
        return False


def send_feedback(transaction_id: str, transaction_data_series: pd.Series, model_pred: int, true_class: int,
                  message: str):
    """Construit le corps de la requête AlertIn et envoie la rétroaction à l'API."""
    try:
        keys_to_drop = ['id', 'model_prediction']
        transaction_dict = transaction_data_series.drop(keys_to_drop, errors='ignore').to_dict()

        transaction_features = {k: float(v) for k, v in transaction_dict.items() if
                                isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())}

        feedback_data = {
            # Les features de transaction sont envoyées directement, l'API s'occupe de la validation Pydantic
            **transaction_features,
            "model_prediction": model_pred,
            "user_feedback": true_class
        }

        if submit_feedback(feedback_data):
            st.success(message)

            # --- CORRECTION DE L'ERREUR JAVASCRIPT : time.sleep(2) EST SUPPRIMÉ ---

            # Suppression de l'alerte de la file d'attente après confirmation réussie
            if 'alerts_queue' in st.session_state and st.session_state.alerts_queue:
                st.session_state.alerts_queue.pop(0)

            get_model_alerts.clear()  # Force la récupération des nouvelles alertes (sans la traitée)
            st.rerun()  # Rechargement immédiat pour passer à l'alerte suivante

        else:
            st.error("Échec de l'enregistrement de la rétroaction")
    except Exception as e:
        st.error(f"Erreur lors de la préparation/envoi de la rétroaction : {e}")


def show():
    """Affiche la page des alertes en temps réel avec des améliorations interactives."""
    load_css()
    apply_button_style()
    create_footer()

    st.title("🚨 Centre de Triage des Alertes")
    st.markdown(
        "Bienvenue dans votre file d'attente d'alertes. Validez les transactions suspectes une par une pour les retirer de la liste.")
    st.markdown("---")

    # Test de connexion à l'API
    try:
        # Test de santé rapide (timeout court)
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            st.error(f"⚠️ L'API ({API_URL}) n'est pas accessible. Statut: {health_response.status_code}")
    except requests.exceptions.RequestException:
        st.error(f"⚠️ Impossible de se connecter à l'API ({API_URL}). Vérifiez la connectivité.")

    # --- Gestion de la file d'attente ---
    if 'alerts_queue' not in st.session_state:
        alerts_df = get_model_alerts()

        if not alerts_df.empty and 'id' not in alerts_df.columns:
            alerts_df['id'] = alerts_df.index.astype(str)
        if alerts_df.empty:
            st.session_state.alerts_queue = []
            st.session_state.initial_alerts_count = 0
        else:
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
        # Conversion en Series et en numérique pour les calculs (V1-V28)
        current_transaction = pd.Series(current_transaction_data).apply(pd.to_numeric, errors='ignore')

        # Déterminer la caractéristique la plus anormale (pour le défaut et l'explication)
        most_anomalous_feature, z_score = find_most_anomalous_feature(current_transaction, historical_df)
        model_verdict = current_transaction.get('model_prediction', 1)

        # --- Affichage des informations clés ---
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col_info_1, col_info_2, col_info_3 = st.columns([1.5, 1, 3])

        with col_info_1:
            st.metric("ID de la transaction", current_transaction['id'])
        with col_info_2:
            amount = current_transaction.get('Amount', 0.0)
            st.metric("Montant", f"{amount:.2f} $")

        with col_info_3:
            if model_verdict == 1:
                st.markdown(f"""
                    <div class='fraud-alert'>
                        <b>Verdict Modèle : SUSPECTÉ DE FRAUDE</b><br>
                        Raison (Heuristique) : {most_anomalous_feature} (Z-score: {z_score:.2f}) est hors norme.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='no-fraud'>
                        <b>Verdict Modèle : NORMAL</b><br>
                        Raison (Heuristique) : {most_anomalous_feature} (Z-score: {z_score:.2f}) est dans la norme.
                    </div>
                """, unsafe_allow_html=True)
                # Tout en bas de la fonction show()
                st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

                # On utilise une clé unique pour chaque page pour éviter les conflits d'ID

                # ou
                logout_button(key="logout_detection")  # Pour la page de détection

        st.markdown("---")

        # --- Visualisation Interactive ---
        st.subheader("Visualisation de l'Anomalie (Analyse de la Distribution)")

        if not historical_df.empty:
            all_v_features = [f"V{i}" for i in range(1, 29)]

            selected_feature = st.selectbox(
                "Choisir la caractéristique PCA à analyser :",
                options=all_v_features,
                index=all_v_features.index(most_anomalous_feature) if most_anomalous_feature in all_v_features else 0,
                key=f"feature_selector_{current_transaction['id']}"
            )

            create_pca_plot(historical_df, current_transaction, feature=selected_feature)

            # --- Affichage des Valeurs PCA sous forme de Tableau ---
        st.markdown("---")
        with st.expander("Voir toutes les valeurs PCA (pour un examen détaillé)"):
            v_data = {k: f"{v:.4f}" if isinstance(v, (int, float)) else v
                      for k, v in current_transaction.items() if k.startswith('V') or k in ['Time', 'Amount']}
            v_df = pd.DataFrame(v_data.items(), columns=['Caractéristique', 'Valeur'])
            st.dataframe(v_df.T, use_container_width=True)

        st.markdown("---")

        # --- Boutons de Rétroaction ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚨 Confirmer FRAUDE (Class=1)", key=f"fraud_{current_transaction['id']}",
                         help="Cliquez pour valider la fraude. La transaction est retirée de la file.", type='primary'):
                with st.spinner("Envoi de la rétroaction..."):
                    send_feedback(current_transaction['id'],
                                  current_transaction,
                                  model_verdict,
                                  1,  # true_class = 1 (Fraude)
                                  "Rétroaction de *fraude* enregistrée avec succès ! Redirection...")
        with col2:
            if st.button("✅ Confirmer NORMAL (Class=0)", key=f"normal_{current_transaction['id']}",
                         help="Cliquez pour confirmer que la transaction est normale. La transaction est retirée de la file.",
                         type='secondary'):
                with st.spinner("Envoi de la rétroaction..."):
                    send_feedback(current_transaction['id'],
                                  current_transaction,
                                  model_verdict,
                                  0,  # true_class = 0 (Normal)
                                  "Rétroaction de transaction *normale* enregistrée avec succès ! Redirection...")

        st.markdown('</div>', unsafe_allow_html=True)

        # Tout en bas de la fonction show() de la page des alertes

        logout_button(key="logout_alerts")  # Une clé unique dédiée à cette page
        # 🌟 Code CSS magique pour coller le bouton tout en bas de la sidebar
        st.sidebar.markdown("""
                <style>
                /* Cible le conteneur de la sidebar pour utiliser Flexbox */
                [data-testid="stSidebarUserContent"] {
                    display: flex;
                    flex-direction: column;
                    min-height: 85vh; /* Aligne sur la hauteur de l'écran */
                }
                /* Pousse le dernier élément (notre bouton) tout en bas */
                [data-testid="stSidebarUserContent"] > div:last-child {
                    margin-top: auto;
                    padding-bottom: 20px;
                }
                </style>
            """, unsafe_allow_html=True)

        # Ton bouton s'affichera maintenant parfaitement en bas
        #logout_button(key="logout_detection")


if __name__ == "__main__":
    show()