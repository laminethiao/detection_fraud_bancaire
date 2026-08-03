import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import numpy as np
from typing import List, Dict, Any


# Modèle pour la validation des données d'entrée
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# Modèle pour la rétroaction (inclut la classe correcte)
class FeedbackData(Transaction):
    # Ceci représente la vérité terrain fournie par l'utilisateur
    user_feedback: int = 0
    # Ajoutez la prédiction du modèle pour stocker l'historique de l'alerte
    model_prediction: int = 0


# Initialisation de l'API FastAPI
app = FastAPI()

# Variables globales pour le modèle, le scaler et les données historiques
model = None
scaler = None
historical_df = pd.DataFrame()  # DataFrame pour stocker les données historiques
# Fichier de stockage pour la rétroaction
FEEDBACK_FILE = "feedback_data.csv"


# --- FONCTIONS DE CHARGEMENT ---

def load_historical_data_for_api():
    """
    CHARGE VOS DONNÉES HISTORIQUES COMPLÈTES ICI.
    (Remplacer le chemin ci-dessous par votre vrai chemin vers creditcard.csv ou autre)
    """
    data_path = os.path.join('app', 'data', 'creditcard.csv')  # REMPLACER PAR VOTRE CHEMIN

    if not os.path.exists(data_path):
        print(
            f"ATTENTION: Fichier de données historiques non trouvé à {data_path}. Les visualisations de la page alertes ne fonctionneront pas.")
        return pd.DataFrame()

    df = pd.read_csv(data_path)

    # Simuler la colonne d'heure pour la compatibilité si votre fichier ne l'a pas
    if 'Hour' not in df.columns and 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(lambda x: pd.to_datetime(x, unit='s').hour)

    return df


@app.on_event("startup")
def load_files_on_startup():
    """
    Charge le modèle, le scaler et les données historiques au démarrage de l'API.
    """
    global model, scaler, historical_df
    try:
        # 1. Chargement Modèle et Scaler
        model_filename = os.path.join('app', 'models', 'xgb_fraud_detection_model.pkl')
        scaler_filename = os.path.join('app', 'models', 'scaler.pkl')

        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)

        # 2. Chargement des données historiques
        historical_df = load_historical_data_for_api()

        print(f"✅ Modèle, Scaler et {len(historical_df)} lignes de données chargées pour l'API.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des fichiers: {e}")
        model = None
        scaler = None


# --- NOUVEAUX ENDPOINTS REQUIS PAR LA PAGE ALERTES ---

@app.get("/health")
async def health_check():
    """
    Endpoint de santé requis par Streamlit (status=200).
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {"status": "ok"}


@app.get("/historical_data")
def load_historical_data_for_api():
    """
    CHARGE VOS DONNÉES HISTORIQUES COMPLÈTES.
    """
    # ⬅️ VÉRIFIEZ QUE LE NOM DU FICHIER EST EXACTEMENT 'credicard_cleaned.csv'
    data_path = os.path.join('app', 'data', 'creditcard_cleaned.csv')

    if not os.path.exists(data_path):
        print(f"ATTENTION: Fichier de données historiques non trouvé à {data_path}.")
        return pd.DataFrame()

    df = pd.read_csv(data_path)
    # ... le reste du code de chargement ...
    return df

# --- ENDPOINTS EXISTANTS CORRIGÉS ---

@app.get("/")
async def home():
    """
    Affiche un message de bienvenue pour confirmer que l'API est en ligne.
    """
    return {
        "message": "API de détection de fraude en cours d'exécution."
    }


# 1. On crée une liste globale en mémoire pour stocker les alertes EN ATTENTE de traitement
# Modifie cette variable dans ton main.py pour injecter des données de test
pending_alerts = [
    {
        "id": "99901",
        "Time": 406.0,
        "Amount": 1250.00,
        "model_prediction": 1,
        "V1": -2.31, "V2": 1.95, "V3": -1.60, "V4": 3.99, "V5": -0.52,
        "V6": -1.42, "V7": -2.53, "V8": 1.39, "V9": -2.77, "V10": -2.77,
        "V11": 3.20, "V12": -2.90, "V13": -0.59, "V14": -4.28, "V15": 0.38,
        "V16": -1.14, "V17": -2.83, "V18": -0.01, "V19": 0.41, "V20": 0.12,
        "V21": 0.51, "V22": -0.03, "V23": -0.46, "V24": 0.32, "V25": 0.04,
        "V26": 0.52, "V27": 0.25, "V28": -0.02
    },
    {
        "id": "99902",
        "Time": 900.0,
        "Amount": 720.50,
        "model_prediction": 1,
        "V1": -1.15, "V2": 0.85, "V3": -2.10, "V4": 2.50, "V5": -0.30,
        "V6": -0.90, "V7": -1.80, "V8": 0.75, "V9": -1.50, "V10": -2.10,
        "V11": 2.10, "V12": -2.20, "V13": -0.10, "V14": -3.50, "V15": 0.20,
        "V16": -0.80, "V17": -2.00, "V18": -0.05, "V19": 0.20, "V20": 0.05,
        "V21": 0.35, "V22": -0.01, "V23": -0.25, "V24": 0.15, "V25": 0.02,
        "V26": 0.30, "V27": 0.10, "V28": -0.01
    }
]


@app.post("/predict")
async def predict_transaction(transaction: Transaction):
    global model, scaler
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Le service est indisponible.")

    try:
        raw_data = transaction.model_dump()
        df = pd.DataFrame([raw_data])
        scaled_features = ['Time', 'Amount']
        df_to_scale = df[scaled_features].copy()
        df[scaled_features] = scaler.transform(df_to_scale)

        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0][1]

        # 🚨 AJOUT CRUCIAL : Si le modèle détecte une fraude (prediction == 1),
        # on l'ajoute dans la liste des alertes EN ATTENTE du Centre de Triage
        if int(prediction) == 1:
            alert_item = raw_data.copy()
            alert_item['model_prediction'] = 1
            # On génère un ID temporaire basé sur la longueur de la liste
            alert_item['id'] = str(len(pending_alerts) + 1)
            pending_alerts.append(alert_item)
            print(f"🚨 Nouvelle alerte ajoutée au Centre de Triage (Total en attente : {len(pending_alerts)})")

        return {
            "prediction": int(prediction),
            "probability": float(prediction_proba),
            "confidence": "Haute" if prediction_proba > 0.8 else ("Moyenne" if prediction_proba > 0.5 else "Basse")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction: {e}")


@app.post("/alert")
def submit_alert_feedback(data: FeedbackData):
    """
    Enregistre la rétroaction manuelle d'un analyste et retire la transaction des alertes en attente.
    """
    try:
        df = pd.DataFrame([data.model_dump()])
        header = not os.path.exists(FEEDBACK_FILE)
        df.to_csv(FEEDBACK_FILE, mode='a', header=header, index=False)
        print(f"✅ Rétroaction enregistrée : {data.model_dump()}")

        # 🚨 AJOUT CRUCIAL : Dès qu'on reçoit une rétroaction, on nettoie la file d'attente
        # On supprime la transaction traitée de la liste `pending_alerts` si elle s'y trouve
        global pending_alerts
        # On compare les montants et le Time pour identifier la transaction traitée
        pending_alerts = [
            a for a in pending_alerts
            if not (abs(a.get('Amount', 0) - data.Amount) < 0.01 and abs(a.get('Time', 0) - data.Time) < 0.01)
        ]

        return {"message": "Rétroaction enregistrée avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de l'enregistrement de la rétroaction : {e}")


@app.get("/alerts")
def get_alerts():
    """
    Renvoie UNIQUEMENT les alertes suspectées par le modèle qui n'ont pas encore été traitées.
    """
    # 🚨 MODIFICATION : Le Centre de Triage lit maintenant les alertes en attente
    # au lieu de lire l'historique des validations passées.
    return {"alerts": pending_alerts}