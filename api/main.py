import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware # 🚨 Nouvelle Importation
import os
import numpy as np
from typing import List, Dict, Any # 🚨 Nouvelle Importation

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

# 🚨 NOUVEAU MODÈLE POUR LE FEEDBACK (AlertIn)
class AlertIn(BaseModel):
    transaction: Transaction
    model_prediction: int # Prédiction initiale du modèle (0 ou 1)
    user_feedback: int    # Classe réelle validée par l'utilisateur (0 ou 1)

# Initialisation de l'API FastAPI
app = FastAPI(title="Fraud Detection API")

# Configuration CORS pour Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variables Globales ---
model = None
scaler = None
# Fichier de stockage pour la rétroaction MLOps (Log permanent)
FEEDBACK_FILE = "feedback_data.csv"
# 🚨 FILE D'ATTENTE DES ALERTES (Queue en mémoire, pour /alerts)
PENDING_ALERTS_DB: List[Dict[str, Any]] = []

# Fonction pour charger les fichiers au démarrage de l'application (Inchangement)
@app.on_event("startup")
def load_model():
    """
    Charge le modèle et le scaler une seule fois au démarrage de l'API.
    """
    global model, scaler
    try:
        # Chemin corrigé pour pointer vers le dossier 'app/models'
        model_filename = os.path.join('app', 'models', 'xgb_fraud_detection_model.pkl')
        scaler_filename = os.path.join('app', 'models', 'scaler.pkl')

        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        print(f"✅ Modèle et Scaler chargés avec succès pour l'API.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des fichiers: {e}")
        model = None
        scaler = None

# Endpoint de base pour vérifier si l'API est en ligne
@app.get("/health")
async def home():
    """
    Vérifie l'état de l'API.
    """
    return {
        "status": "ok",
        "message": "API de détection de fraude en cours d'exécution."
    }

# Endpoint de prédiction (MODIFIÉ pour ajouter à la queue)
@app.post("/predict")
async def predict_transaction(transaction: Transaction):
    """
    Prédit si une transaction est frauduleuse (1) ou non (0).
    Ajoute les alertes à la file PENDING_ALERTS_DB.
    """
    global model, scaler, PENDING_ALERTS_DB

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Le service est indisponible.")

    try:
        df = pd.DataFrame([transaction.model_dump()])
        scaled_features = ['Time', 'Amount']
        df[scaled_features] = scaler.transform(df[scaled_features])

        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0][1]
        
        confidence = "Haute" if prediction_proba > 0.8 else ("Moyenne" if prediction_proba > 0.5 else "Basse")

        # 🚨 LOGIQUE D'ALERTE : Si c'est une fraude, l'ajouter à la file d'attente
        if prediction == 1:
            alert_entry = transaction.model_dump()
            alert_entry['model_prediction'] = int(prediction) # Nécessaire pour le front-end /alert
            alert_entry['prediction_score'] = float(prediction_proba)
            PENDING_ALERTS_DB.append(alert_entry)

        return {
            "prediction": int(prediction),
            "probability": float(prediction_proba),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction: {e}")

# Endpoint pour la rétroaction (NOUVEL ENDPOINT /alert)
@app.post("/alert")
def record_alert_feedback(alert_data: AlertIn):
    """
    Enregistre le feedback de l'utilisateur (MLOps) et retire l'alerte de la file d'attente.
    """
    global PENDING_ALERTS_DB
    
    # 1. Enregistrer les données de feedback (MLOps Log)
    try:
        # Créer le DataFrame avec toutes les informations
        transaction_df = pd.DataFrame([alert_data.transaction.model_dump()])
        # Ajouter les colonnes de MLOps
        transaction_df['model_prediction'] = alert_data.model_prediction
        transaction_df['user_feedback'] = alert_data.user_feedback
        
        # Écrire dans le fichier CSV (log permanent)
        header = not os.path.exists(FEEDBACK_FILE)
        transaction_df.to_csv(FEEDBACK_FILE, mode='a', header=header, index=False)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de l'enregistrement de la rétroaction MLOps : {e}")

    # 2. Retirer l'alerte de la file d'attente PENDING_ALERTS_DB
    tx_time = alert_data.transaction.Time
    tx_amount = alert_data.transaction.Amount
    
    # On filtre la file d'attente pour exclure la transaction traitée
    new_alerts_db = [
        alert for alert in PENDING_ALERTS_DB 
        if not (alert.get('Time') == tx_time and alert.get('Amount') == tx_amount)
    ]
    
    # Mise à jour de la queue globale
    PENDING_ALERTS_DB = new_alerts_db

    return {"status": "success", "message": "Feedback enregistré et alerte retirée de la file."}


# Endpoint pour l'historique des alertes - (MODIFIÉ pour lire la queue en mémoire)
@app.get("/alerts")
def get_alerts():
    """
    Récupère la liste des alertes de fraude non résolues (depuis la queue en mémoire).
    """
    # 🚨 Retourne la queue d'alertes en mémoire, pas le fichier CSV de log complet
    return {"alerts": PENDING_ALERTS_DB}