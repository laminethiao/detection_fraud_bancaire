import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import numpy as np

# Import des fonctions de style à partir d'un autre fichier
# 🚨 Assurez-vous que ces fichiers (dans utils/) sont aussi présents dans votre dépôt GitHub
from utils.ui_style import setup_page_config, load_css, create_footer, apply_button_style
from utils.auth import check_authentication

check_authentication()

# 1. 🔑 MODIFICATION CLÉ : URL de l'API déployée
API_URL = "https://lamine-th0101-detection-fraud-bancaire-api.hf.space"

# 🚨 Nouveaux endpoints basés sur l'API déployée
ALERT_URL = f"{API_URL}/alert"
GET_ALERTS_URL = f"{API_URL}/alerts"
HISTORICAL_DATA_URL = f"{API_URL}/historical_data"


@st.cache_data(ttl=5)
def get_model_alerts():
    """
    Récupère la liste des alertes de fraude à partir de l'API.
    """
    try:
        # 🚨 Utilisation de l'endpoint d'API déployé
        response = requests.get(GET_ALERTS_URL, timeout=10)
        if response.status_code == 200:
            alerts_data = response.json().get('alerts', [])
            if not alerts_data:
                return pd.DataFrame()
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df['id'] = alerts_df.index.astype(str)
            return alerts_df
        else:
            st.error(f"Erreur lors de la récupération des alertes : {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de se connecter à l'API ({GET_ALERTS_URL}). Vérifiez l'état de l'API. Erreur: {e}")
        return pd.DataFrame()

# 2. 🔑 MODIFICATION CLÉ : Retrait du chargement local du fichier CSV
@st.cache_data
def get_historical_data():
    """
    Récupère des données historiques pour la visualisation UNIQUEMENT via l'API.
    Ceci est nécessaire pour le déploiement en ligne.
    """
    with st.spinner("Chargement des données historiques..."):
        try:
            # Tente de récupérer les données depuis l'API déployée
            response = requests.get(HISTORICAL_DATA_URL, timeout=10)
            if response.status_code == 200:
                data = response.json().get('data', [])
                return pd.DataFrame(data)
            else:
                # Échec de l'API
                st.error(f"Erreur lors de la récupération des données historiques depuis l'API : {response.status_code}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            # Échec de la connexion à l'API
            st.error(f"Erreur de connexion à l'API pour les données historiques. Assurez-vous que l'API est saine. Erreur: {e}")
            return pd.DataFrame()

def find_most_anomalous_feature(current_transaction, historical_df):
    """
    Trouve la caractéristique (V1-V28) qui est le plus en dehors de la distribution normale.
    Retourne la caractéristique et son Z-score. (Logique inchangée)
    """
    if historical_df.empty:
        return 'V1', 0.0

    anomalies = {}
    # S'assurer que 'Class' existe avant le filtrage. L'API doit la fournir.
    if 'Class' not in historical_df.columns:
        st.warning("La colonne 'Class' est manquante dans les données historiques de l'API.")
        return 'V1', 0.0

    normal_data = historical_df[historical_df['Class'] == 0]

    for feature in [f"V{i}" for i in range(1, 29)]:
        if feature in normal_data.columns and feature in current_transaction:
            mean = normal_data[feature].mean()
            std = normal_data[feature].std()

            if std > 0:
                # Assurez-vous que la valeur de la transaction est un float avant la soustraction
                try:
                    current_value = float(current_transaction[feature])
                except ValueError:
                    continue # Skip if value is not convertible to float

                z_score = abs(current_value - mean) / std
                anomalies[feature] = z_score

    if anomalies:
        most_anomalous_feature = max(anomalies, key=anomalies.get)
        return most_anomalous_feature, anomalies[most_anomalous_feature]
    else:
        return 'V1', 0.0

def create_pca_plot(df, current_transaction, feature):
    """Crée un graphique de distribution pour une valeur PCA. (Logique inchangée)"""
    fig = go.Figure()

    if df.empty or 'Class' not in df.columns or feature not in df.columns:
        st.error("Impossible de créer le graphique car les données historiques ou la colonne 'Class' sont manquantes.")
        return

    normal_data = df[df['Class'] == 0]
    fraud_data = df[df['Class'] == 1]

    # Générer la courbe de densité pour les transactions normales
    if not normal_data.empty and len(normal_data[feature].unique()) > 1: # Nécessite au moins 2 points pour kde
        try:
            kde = gaussian_kde(normal_data[feature].dropna())
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
        except ValueError:
            # Gérer le cas où kde échoue (e.g., données non numériques ou trop peu de points)
            pass

    # Ajouter l'histogramme des transactions frauduleuses (pour référence)
    if not fraud_data.empty:
        fig.add_trace(go.Histogram(
            x=fraud_data[feature],
            name='Transactions Frauduleuses',
            marker_color='#dc3545',
            opacity=0.6,
            histnorm='probability density'
        ))

    # Ajouter la ligne de la transaction actuelle
    try:
        current_value = float(current_transaction[feature])
        fig.add_vline(
            x=current_value,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Transaction actuelle: {current_value:.2f}",
            annotation_position="top right"
        )
    except (ValueError, KeyError):
        pass # Ignorer si la valeur n'est pas un nombre ou n'existe pas

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
    Soumet une rétroaction à l'API en utilisant l'endpoint /alert.
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
        st.error(f"Erreur de connexion à l'API lors de l'envoi de feedback: {e}")
        return False

def send_feedback(transaction_id, transaction_data_series, model_pred: int, true_class: int, message: str):
    """
    Construit le corps de la requête AlertIn et envoie la rétroaction à l'API.
    """
    try:
        # Convertir les données de la transaction du format Series au dictionnaire float
        # Exclure les colonnes de métadonnées qui ne sont pas des features de transaction
        keys_to_drop = ['id', 'model_prediction']
        transaction_dict = transaction_data_series.drop(keys_to_drop, errors='ignore').to_dict()
        
        # Le endpoint /alert nécessite que toutes les valeurs numériques soient des floats
        transaction_features = {k: float(v) for k, v in transaction_dict.items() if isinstance(v, (int, float, str))}
        
        feedback_data = {
            "transaction": transaction_features,
            "model_prediction": model_pred,
            "user_feedback": true_class
        }
        
        if submit_feedback(feedback_data):
            st.success(message)
            time.sleep(2)
            # Suppression de l'alerte de la file d'attente après confirmation réussie
            if 'alerts_queue' in st.session_state and st.session_state.alerts_queue:
                st.session_state.alerts_queue.pop(0)
            # Invalider le cache de la fonction get_model_alerts()
            get_model_alerts.clear() 
            st.rerun()
        else:
            st.error("Échec de l'enregistrement de la rétroaction")
    except Exception as e:
        st.error(f"Erreur lors de la préparation/envoi de la rétroaction : {e}")

def show():
    """Affiche la page des alertes en temps réel avec des améliorations interactives."""
    # setup_page_config() # Décommenter si vous utilisez setup_page_config
    load_css()
    apply_button_style()
    create_footer()

    st.title("🚨 Centre de Triage des Alertes")
    st.markdown(
        "Bienvenue dans votre file d'attente d'alertes. Validez les transactions suspectes une par une pour les retirer de la liste.")
    st.markdown("---")

    # Test de connexion à l'API
    try:
        health_response = requests.get(f"{API_URL}/health", timeout=5)
        if health_response.status_code != 200:
            st.error(f"⚠️ L'API ({API_URL}) n'est pas accessible. Statut: {health_response.status_code}")
    except:
        st.error(f"⚠️ Impossible de se connecter à l'API ({API_URL}). Vérifiez la connectivité.")

    # --- Gestion de la file d'attente ---
    if 'alerts_queue' not in st.session_state:
        alerts_df = get_model_alerts()
        
        required_cols = alerts_df.columns.tolist() 
        if 'id' not in required_cols:
            alerts_df['id'] = alerts_df.index.astype(str)
                 
        if 'model_prediction' not in required_cols:
            # Par défaut, une alerte est une prédiction de fraude (1)
            alerts_df['model_prediction'] = 1 
            
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
        # Convertir en Series, en s'assurant que les colonnes numériques sont au bon format
        current_transaction = pd.Series(current_transaction_data).apply(pd.to_numeric, errors='ignore')

        # Déterminer la caractéristique la plus anormale (pour le défaut et l'explication)
        most_anomalous_feature, z_score = find_most_anomalous_feature(current_transaction, historical_df)
        
        # Le verdict du modèle est pris directement de la donnée d'alerte si disponible,
        model_verdict = current_transaction.get('model_prediction', 1) 
        
        # --- Affichage des informations clés ---
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col_info_1, col_info_2, col_info_3 = st.columns([1.5, 1, 3])

        with col_info_1:
            st.metric("ID de la transaction", current_transaction['id'])
        with col_info_2:
            # S'assurer que 'Amount' est un nombre
            amount = current_transaction.get('Amount', 0.0)
            st.metric("Montant", f"{amount:.2f} $")

        with col_info_3:
            # Afficher l'explication du modèle
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
            # Exclure les colonnes non-transactionnelles
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
                                  current_transaction.drop(['id', 'model_prediction'], errors='ignore'), 
                                  model_verdict, 
                                  1,
                                  "Rétroaction de *fraude* enregistrée avec succès ! Redirection...")
        with col2:
            if st.button("✅ Confirmer NORMAL (Class=0)", key=f"normal_{current_transaction['id']}",
                         help="Cliquez pour confirmer que la transaction est normale. La transaction est retirée de la file.",
                         type='secondary'):
                with st.spinner("Envoi de la rétroaction..."):
                    send_feedback(current_transaction['id'], 
                                  current_transaction.drop(['id', 'model_prediction'], errors='ignore'), 
                                  model_verdict, 
                                  0,
                                  "Rétroaction de transaction *normale* enregistrée avec succès ! Redirection...")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    show()