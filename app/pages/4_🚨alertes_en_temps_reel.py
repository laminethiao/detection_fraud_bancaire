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
API_URL = "https://lamine-th0101-detection-fraud-bancaire-api.hf.space"
# 🚨 NOUVEL ENDPOINT DÉDIÉ AUX ALERTES ET AU FEEDBACK
ALERT_URL = f"{API_URL}/alert"
# L'endpoint de récupération d'alertes reste le même
GET_ALERTS_URL = f"{API_URL}/alerts"
# L'endpoint des données historiques reste le même
HISTORICAL_DATA_URL = f"{API_URL}/historical_data"


@st.cache_data(ttl=5)
def get_model_alerts():
    """
    Récupère la liste des alertes de fraude à partir de l'API.
    La fonction est mise en cache pour 5 secondes pour un rafraîchissement régulier.
    """
    try:
        # 🚨 Utilisation de la variable GET_ALERTS_URL
        response = requests.get(GET_ALERTS_URL, timeout=10)
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
        st.error(f"Impossible de se connecter à l'API : {e}")
        return pd.DataFrame()

@st.cache_data
def get_historical_data():
    """
    Récupère des données historiques pour la visualisation.
    """
    file_path = "data/creditcard_cleaned.csv"

    with st.spinner("Chargement des données historiques..."):
        try:
            # 🚨 Utilisation de la variable HISTORICAL_DATA_URL
            response = requests.get(HISTORICAL_DATA_URL, timeout=5)
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
    Retourne la caractéristique et son Z-score. (Logique inchangée)
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
        return most_anomalous_feature, anomalies[most_anomalies_feature]
    else:
        return 'V1', 0.0

def create_pca_plot(df, current_transaction, feature):
    """Crée un graphique de distribution pour une valeur PCA. (Logique inchangée)"""
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

def submit_feedback(feedback_data):
    """
    Soumet une rétroaction à l'API en utilisant le nouvel endpoint /alert.
    """
    try:
        # 🚨 Utilisation de l'endpoint /alert pour le feedback
        response = requests.post(ALERT_URL, json=feedback_data, timeout=10)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Erreur lors de l'envoi de la rétroaction: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return False

def send_feedback(transaction_id, transaction_data_series, model_pred: int, true_class: int, message: str):
    """
    Construit le corps de la requête AlertIn et envoie la rétroaction à l'API.
    """
    try:
        # Convertir les données de la transaction du format Series au dictionnaire float
        transaction_dict = transaction_data_series.to_dict()
        
        # Le endpoint /alert nécessite que toutes les valeurs numériques soient des floats
        # et que le corps soit au format AlertIn.
        feedback_data = {
            "transaction": {k: float(v) for k, v in transaction_dict.items()},
            "model_prediction": model_pred,
            "user_feedback": true_class
        }
        
        if submit_feedback(feedback_data):
            st.success(message)
            time.sleep(2)
            # Suppression de l'alerte de la file d'attente après confirmation réussie
            if 'alerts_queue' in st.session_state and st.session_state.alerts_queue:
                st.session_state.alerts_queue.pop(0)
            # 💡 Mise à jour : Il est plus sûr d'invalider le cache de la fonction get_model_alerts()
            # pour forcer le rafraîchissement des alertes disponibles
            get_model_alerts.clear() 
            st.rerun()
        else:
            st.error("Échec de l'enregistrement de la rétroaction")
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de la rétroaction : {e}")

def show():
    """Affiche la page des alertes en temps réel avec des améliorations interactives."""
    load_css()
    apply_button_style()
    create_footer()

    st.title("🚨 Centre de Triage des Alertes")
    st.markdown(
        "Bienvenue dans votre file d'attente d'alertes. Validez les transactions suspectes une par une pour les retirer de la liste.")
    st.markdown("---")

    # Test de connexion à l'API (inchangé)
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            st.error("⚠️ L'API n'est pas accessible. Vérifiez que le serveur est en cours d'exécution.")
    except:
        st.error("⚠️ Impossible de se connecter à l'API. Vérifiez votre connexion.")

    # --- Gestion de la file d'attente ---
    if 'alerts_queue' not in st.session_state:
        alerts_df = get_model_alerts()
        # S'assurer que les colonnes 'id', 'Class', 'prediction_score' sont présentes avant de convertir en dict
        # Même si l'API ne renvoie pas 'Class', le front-end le gère
        required_cols = alerts_df.columns.tolist() 
        if 'id' not in required_cols:
             alerts_df['id'] = alerts_df.index.astype(str)
             
        # Dans un scénario réel, l'API devrait retourner la 'model_prediction' et le 'prediction_score'.
        # Nous utilisons la valeur par défaut pour l'exemple.
        if 'model_prediction' not in required_cols:
             alerts_df['model_prediction'] = 1 # Par défaut, une alerte est une prédiction de fraude
        
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
        
        # Le verdict du modèle est pris directement de la donnée d'alerte si disponible,
        # sinon on utilise une heuristique ou la valeur par défaut (1=Fraude)
        model_verdict = current_transaction.get('model_prediction', 1) 
        
        # --- Affichage des informations clés ---
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col_info_1, col_info_2, col_info_3 = st.columns([1.5, 1, 3])

        with col_info_1:
            st.metric("ID de la transaction", current_transaction['id'])
        with col_info_2:
            st.metric("Montant", f"{current_transaction['Amount']:.2f} $")

        with col_info_3:
            # Afficher l'explication du modèle
            if model_verdict == 1: # Utilisation du verdict du modèle
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

        st.markdown("---")

        # --- Visualisation Interactive --- (Logique inchangée)
        st.subheader("Visualisation de l'Anomalie (Analyse de la Distribution)")

        if not historical_df.empty:
            all_v_features = [f"V{i}" for i in range(1, 29)]

            selected_feature = st.selectbox(
                "Choisir la caractéristique PCA à analyser :",
                options=all_v_features,
                index=all_v_features.index(most_anomalous_feature),
                key=f"feature_selector_{current_transaction['id']}"
            )

            create_pca_plot(historical_df, current_transaction, feature=selected_feature)

        # --- Affichage des Valeurs PCA sous forme de Tableau --- (Logique inchangée)
        st.markdown("---")
        with st.expander("Voir toutes les valeurs PCA (pour un examen détaillé)"):
            # Exclure les colonnes non-transactionnelles comme 'id' et 'model_prediction'
            v_data = {k: v for k, v in current_transaction.items() if k.startswith('V') or k in ['Time', 'Amount']}
            v_df = pd.DataFrame(v_data.items(), columns=['Caractéristique', 'Valeur'])
            st.dataframe(v_df.T, use_container_width=True)

        st.markdown("---")

        # --- Boutons de Rétroaction ---
        # 🚨 Mise à jour de l'appel de send_feedback
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚨 Confirmer FRAUDE (Class=1)", key=f"fraud_{current_transaction['id']}",
                         help="Cliquez pour valider la fraude. La transaction est retirée de la file.", type='primary'):
                with st.spinner("Envoi de la rétroaction..."):
                    # 🚨 Passage du model_verdict (la prédiction du modèle) et de la true_class (1)
                    send_feedback(current_transaction['id'], 
                                  current_transaction.drop(['id', 'model_prediction'], errors='ignore'), 
                                  model_verdict, 
                                  1,
                                  "Rétroaction de *fraude* enregistrée avec succès ! Redirection...")
        with col2:
            if st.button("✅ Confirmer NORMAL (Class=0)", key=f"normal_{current_transaction['id']}",
                         help="Cliquez pour confirmer que la transaction est normale. La transaction est retirée de la file.",
                         type='secondary'):
                with st.spinner("Envoi de la rétroaction..."):
                    # 🚨 Passage du model_verdict (la prédiction du modèle) et de la true_class (0)
                    send_feedback(current_transaction['id'], 
                                  current_transaction.drop(['id', 'model_prediction'], errors='ignore'), 
                                  model_verdict, 
                                  0,
                                  "Rétroaction de transaction *normale* enregistrée avec succès ! Redirection...")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()