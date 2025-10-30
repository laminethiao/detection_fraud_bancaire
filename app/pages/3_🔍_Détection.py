import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from typing import Optional

from utils.auth import check_authentication

# Assurez-vous d'avoir les fonctions ui_style importées
try:
    from utils.ui_style import setup_page_config, load_css, create_footer, create_header
    from utils.ui_style import apply_button_style
    UI_STYLE_EXISTS = True
except ImportError:
    UI_STYLE_EXISTS = False

check_authentication()

# URL de l'API FastAPI
API_URL = "http://detection-fraud-bancaire.fly.dev"
PREDICT_URL = f"{API_URL}/predict"
FEEDBACK_URL = f"{API_URL}/feedback"

# Exemple de transaction (classe = 0, non-fraude)
TRANSACTION_EXAMPLE = {
    "Time": 0.0, "V1": -1.3598071336738, "V2": -0.0727811733098497, "V3": 2.53634673796914,
    "V4": 1.37815522427443, "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507, "V9": 0.363786969611213,
    "V10": 0.0907941719789316, "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879, "V15": 1.46817697209427,
    "V16": -0.470400525259478, "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705, "V21": -0.018306777944153,
    "V22": 0.277837575558899, "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824, "V27": 1.33558376740387e-01,
    "V28": -0.0210530534538215, "Amount": 149.62
}

# Exemple de transaction (classe = 1, fraude)
TRANSACTION_FRAUD_EXAMPLE = {
    "Time": 406.0, "V1": -2.3122265423263, "V2": 1.95199201150017, "V3": -1.60985072049533,
    "V4": 3.99790558832009, "V5": -0.522187864274941, "V6": -1.42654531920537,
    "V7": -2.53738730624021, "V8": 1.39165725068481, "V9": -2.77008927712437,
    "V10": -2.7722721446714, "V11": 3.20203302017502, "V12": -2.89990738849947,
    "V13": -0.595221881324605, "V14": -4.28925424754593, "V15": 0.38972412089012,
    "V16": -1.14074717981966, "V17": -2.83005567450419, "V18": -0.0168224684077754,
    "V19": 0.416955705007305, "V20": 0.126910549495066, "V21": 0.517232370866083,
    "V22": -0.0350493686053065, "V23": -0.465211075723555, "V24": 0.320198198514521,
    "V25": 0.04403362024523, "V26": 0.525940409896627, "V27": 0.251105009132223,
    "V28": -0.0210530534538215, "Amount": 0.0
}

def predict_transaction(transaction_data):
    """
    Prédit si une transaction est frauduleuse via l'API.
    """
    try:
        response = requests.post(PREDICT_URL, json=transaction_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

def submit_feedback(feedback_data):
    """
    Soumet une rétroaction à l'API.
    """
    try:
        response = requests.post(FEEDBACK_URL, json=feedback_data, timeout=10)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Erreur lors de l'envoi de la rétroaction: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return False

def show():
    """
    Affiche la page de détection de fraude en temps réel.
    """

    # Initialisation de l'état de session
    if 'Time' not in st.session_state:
        st.session_state['Time'] = TRANSACTION_EXAMPLE['Time']
    if 'Amount' not in st.session_state:
        st.session_state['Amount'] = TRANSACTION_EXAMPLE['Amount']

    # Initialiser les V-features
    for i in range(1, 29):
        key = f"V{i}"
        if key not in st.session_state:
            st.session_state[key] = TRANSACTION_EXAMPLE.get(key, 0.0)

    # Initialisation des clés de résultats
    if 'last_prediction_class' not in st.session_state:
        st.session_state['last_prediction_class'] = None
    if 'last_transaction_data' not in st.session_state:
        st.session_state['last_transaction_data'] = None

    if UI_STYLE_EXISTS:
        load_css()
        apply_button_style()
        create_footer()

    st.title("🔍 Détection en Temps Réel")
    st.markdown("""
    Utilisez ce formulaire pour tester notre modèle de détection de fraude sur une transaction individuelle.
    Saisissez les paramètres de la transaction ou utilisez les exemples fournis.
    """)

    # Boutons pour charger les données d'exemples
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Charger un exemple non-fraude"):
            for key, value in TRANSACTION_EXAMPLE.items():
                st.session_state[key] = value
            st.session_state['last_transaction_data'] = TRANSACTION_EXAMPLE

    with col2:
        if st.button("Charger un exemple de fraude"):
            for key, value in TRANSACTION_FRAUD_EXAMPLE.items():
                st.session_state[key] = value
            st.session_state['last_transaction_data'] = TRANSACTION_FRAUD_EXAMPLE

    # Formulaire de saisie des données
    with st.form("transaction_form"):
        st.subheader("Entrez les détails de la transaction")

        col1, col2 = st.columns(2)
        with col1:
            time_val = st.session_state.get("Time")
            time = st.number_input("Time", value=time_val, step=0.01, format="%.2f",
                                   help="Temps écoulé depuis la première transaction en secondes.")
        with col2:
            amount_val = st.session_state.get("Amount")
            amount = st.number_input("Amount", value=amount_val, step=0.01, format="%.2f",
                                     help="Montant de la transaction.")

        st.markdown("---")
        st.subheader("Variables anonymisées (V1-V28)")

        # Création des colonnes pour un affichage propre des V-features
        cols_v = st.columns(4)
        v_features = {}
        for i in range(1, 29):
            col_index = (i - 1) % 4
            with cols_v[col_index]:
                v_feature_val = st.session_state.get(f"V{i}")
                v_features[f"V{i}"] = st.number_input(f"V{i}", value=v_feature_val, step=0.01, format="%.2f")

        submit_button = st.form_submit_button(label="Analyser la transaction")

    # Traitement de la soumission du formulaire
    if submit_button:
        transaction_data = {
            "Time": time,
            "Amount": amount,
            **v_features
        }

        # Appel à l'API pour la prédiction
        with st.spinner("🔍 Analyse de la transaction en cours..."):
            prediction_result = predict_transaction(transaction_data)

        if prediction_result:
            st.session_state['last_transaction_data'] = transaction_data
            st.session_state['last_prediction_class'] = prediction_result.get("prediction")
            st.session_state['last_prediction_prob'] = prediction_result.get("probability", 0.0)
            st.session_state['last_prediction_confidence'] = prediction_result.get("confidence", "Non disponible")
        else:
            st.error("❌ Impossible d'obtenir une prédiction de l'API.")

    # Affichage des résultats
    if 'last_prediction_class' in st.session_state and st.session_state.get('last_prediction_class') is not None:
        st.markdown("---")
        st.subheader("Résultat de la prédiction")

        prediction_class = st.session_state.get('last_prediction_class')
        prediction_prob = st.session_state.get('last_prediction_prob', 0.0)
        prediction_confidence = st.session_state.get('last_prediction_confidence', "Non disponible")

        if prediction_class == 1:
            st.error("🚨 La transaction est **suspectée de fraude** !")
        elif prediction_class == 0:
            st.success("✅ La transaction est **normale**.")
        else:
            st.warning("⚠️ Prédiction non disponible ou invalide.")

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Probabilité de fraude", f"{prediction_prob:.2%}")
        with col_res2:
            st.metric("Niveau de confiance", prediction_confidence)

        st.info(
            "💡 Remarque : La prédiction se base sur le modèle XGBoost, mais une vérification manuelle peut être nécessaire pour les cas ambigus.")

        # Section Recommandations et Facteurs Clés
        st.markdown("---")
        with st.expander("Recommandations et Facteurs Clés"):
            # Données fictives pour l'importance des caractéristiques
            fake_importance = {
                'V17': 0.25, 'V14': 0.20, 'V12': 0.15, 'V10': 0.12, 'V11': 0.08, 'Amount': 0.05
            }
            importance_df = pd.DataFrame(
                list(fake_importance.items()), columns=['Variable', 'Importance']
            ).sort_values(by='Importance', ascending=True)

            if prediction_class == 1:
                st.warning(
                    "Cette transaction présente des signes de fraude. Voici pourquoi et ce qu'il faut faire :")
                st.markdown("""
                * **Bloquer la transaction :** Pour prévenir toute perte financière immédiate.
                * **Contacter le client :** Pour vérifier si l'achat est légitime ou non.
                * **Marquer le compte :** Pour une surveillance renforcée et l'analyse d'activités futures.
                """)
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Variable',
                    orientation='h',
                    title='Variables influençant le plus la prédiction de fraude',
                    color_discrete_sequence=px.colors.sequential.Reds_r
                )
                st.plotly_chart(fig, use_container_width=True)

            elif prediction_class == 0:
                st.info("Cette transaction est considérée comme sûre. Voici les actions et les facteurs clés :")
                st.markdown("""
                * **Traiter la transaction :** La transaction est validée et peut être complétée.
                * **Surveillance habituelle :** Le compte ne nécessite pas de surveillance supplémentaire pour le moment.
                * **Encourager l'utilisation :** Une transaction fluide renforce la confiance du client.
                """)
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Variable',
                    orientation='h',
                    title='Variables renforçant la prédiction de non-fraude',
                    color_discrete_sequence=px.colors.sequential.Greens_r
                )
                st.plotly_chart(fig, use_container_width=True)

        # Section de rétroaction pour l'analyste
        st.markdown("---")
        st.subheader("Confirmation manuelle de la transaction")

        if prediction_class == 0:
            st.markdown("Le modèle a prédit que cette transaction est **normale**. Veuillez confirmer ce résultat.")
            if st.button("✅ Confirmer comme normale", key="confirm_non_fraud"):
                feedback_data = st.session_state['last_transaction_data'].copy()
                feedback_data['Class'] = 0
                if submit_feedback(feedback_data):
                    st.success("🎉 Rétroaction enregistrée avec succès : Transaction marquée comme non-fraude.")
                else:
                    st.error("❌ Échec de l'enregistrement de la rétroaction.")

        elif prediction_class == 1:
            st.markdown("Le modèle a prédit que cette transaction est **frauduleuse**. Veuillez confirmer ce résultat.")
            if st.button("🚨 Confirmer comme fraude", key="confirm_fraud"):
                feedback_data = st.session_state['last_transaction_data'].copy()
                feedback_data['Class'] = 1
                if submit_feedback(feedback_data):
                    st.success("🎉 Rétroaction enregistrée avec succès : Transaction marquée comme fraude.")
                else:
                    st.error("❌ Échec de l'enregistrement de la rétroaction.")

if __name__ == "__main__":
    show()