import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from typing import List, Dict, Any

# --- MODÈLES DE DONNÉES (SCHEMAS) ---

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

class AlertIn(BaseModel):
    transaction: Transaction
    model_prediction: int
    user_feedback: int

class BatchTransactions(BaseModel):
    transactions: List[Transaction] # Requis par Dashbord.py

# --- INITIALISATION ET CONFIGURATION ---

app = FastAPI(title="Fraud Detection API")

# Configuration CORS pour Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- VARIABLES GLOBALES ET DONNÉES ---
model = None
scaler = None
FEEDBACK_FILE = "feedback_data.csv"
PENDING_ALERTS_DB: List[Dict[str, Any]] = []

# --- FONCTIONS DE CHARGEMENT ---

@app.on_event("startup")
def load_model():
    """Charge le modèle et le scaler au démarrage de l'API."""
    global model, scaler
    try:
        model_filename = os.path.join('app', 'models', 'xgb_fraud_detection_model.pkl')
        scaler_filename = os.path.join('app', 'models', 'scaler.pkl')

        model = joblib.load(model_filename)
        scaler = joblib.load(scaler_filename)
        print("✅ Modèle et Scaler chargés.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des fichiers: {e}")
        model = None
        scaler = None

def load_historical_data_df():
    """Charge un échantillon du DataFrame historique."""
    try:
        # 🚨 ASSUREZ-VOUS QUE CE CHEMIN EST CORRECT DANS VOTRE DÉPÔT GITHUB
        file_path = "data/creditcard_cleaned.csv" 
        df = pd.read_csv(file_path).sample(n=10000, random_state=42)
        if 'Class' not in df.columns:
             df['Class'] = 0 # Fallback si Class est manquante
        return df
    except FileNotFoundError:
        print(f"❌ Fichier historique non trouvé : {file_path}")
        return pd.DataFrame()

# --- ENDPOINTS D'ÉTAT ET DE DONNÉES ---

@app.get("/health")
async def get_health():
    """Vérification de l'état de l'API."""
    return {"status": "ok", "message": "API de détection de fraude en cours d'exécution."}

@app.get("/historical_data")
async def get_historical_data():
    """Fournit des données historiques pour la visualisation Streamlit (Dashbord et Alertes)."""
    df = load_historical_data_df()
    if df.empty:
        raise HTTPException(status_code=500, detail="Impossible de charger les données historiques côté API.")
        
    return {"data": df.to_dict(orient="records")}

@app.get("/alerts")
def get_alerts():
    """Récupère la liste des alertes de fraude non résolues (queue en mémoire)."""
    return {"alerts": PENDING_ALERTS_DB}

# --- ENDPOINTS DE PRÉDICTION ---

@app.post("/predict")
async def predict_transaction(transaction: Transaction):
    """Prédit une seule transaction (utilisé par Detection.py)."""
    global model, scaler, PENDING_ALERTS_DB

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    try:
        df = pd.DataFrame([transaction.model_dump()])
        scaled_features = ['Time', 'Amount']
        df[scaled_features] = scaler.transform(df[scaled_features])

        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0][1]
        confidence = "Haute" if prediction_proba > 0.8 else ("Moyenne" if prediction_proba > 0.5 else "Basse")

        # LOGIQUE D'ALERTE : Ajouter à la file d'attente si fraude
        if prediction == 1:
            alert_entry = transaction.model_dump()
            alert_entry['model_prediction'] = int(prediction)
            alert_entry['prediction_score'] = float(prediction_proba)
            PENDING_ALERTS_DB.append(alert_entry)

        return {
            "prediction": int(prediction),
            "probability": float(prediction_proba),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction: {e}")

@app.post("/predict_batch")
async def predict_batch(batch_data: BatchTransactions):
    """Prédit un lot de transactions (utilisé par Dashbord.py)."""
    global model, scaler, PENDING_ALERTS_DB
    
    # 1. Préparer les données en DataFrame
    df = pd.DataFrame([t.model_dump() for t in batch_data.transactions])
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")

    try:
        # 2. Normaliser les variables 'Time' et 'Amount'
        scaled_features = ['Time', 'Amount']
        df[scaled_features] = scaler.transform(df[scaled_features])

        # 3. Faire la prédiction par lot
        predictions = model.predict(df)
        prediction_probas = model.predict_proba(df)[:, 1]
        
        # 4. Gérer les Alertes : Ajouter à la file d'attente (logique simplifiée)
        for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
            if pred == 1:
                # 🚨 N'ajoutez que les nouvelles transactions à la file,
                # mais dans le contexte du Dashboard, on se contente de la prédiction.
                # L'ajout en masse ici peut être lourd. Si vous voulez un comportement
                # plus précis, il faudrait un mécanisme de vérification d'unicité.
                pass 
                
        # 5. Retourner la liste des prédictions
        return {"predictions": [int(p) for p in predictions]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction par lot: {e}")


# --- ENDPOINT DE FEEDBACK (MLOPS) ---

@app.post("/alert")
def record_alert_feedback(alert_data: AlertIn):
    """Enregistre le feedback (MLOps) et retire l'alerte de la queue."""
    global PENDING_ALERTS_DB
    
    # 1. Enregistrer les données de feedback (MLOps Log)
    try:
        transaction_df = pd.DataFrame([alert_data.transaction.model_dump()])
        transaction_df['model_prediction'] = alert_data.model_prediction
        transaction_df['user_feedback'] = alert_data.user_feedback
        
        header = not os.path.exists(FEEDBACK_FILE)
        transaction_df.to_csv(FEEDBACK_FILE, mode='a', header=header, index=False)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de l'enregistrement de la rétroaction MLOps : {e}")

    # 2. Retirer l'alerte de la file d'attente PENDING_ALERTS_DB
    tx_time = alert_data.transaction.Time
    tx_amount = alert_data.transaction.Amount
    
    # Filtrer la file d'attente pour exclure la transaction traitée
    new_alerts_db = [
        alert for alert in PENDING_ALERTS_DB 
        if not (alert.get('Time') == tx_time and alert.get('Amount') == tx_amount)
    ]
    
    PENDING_ALERTS_DB = new_alerts_db

    return {"status": "success", "message": "Feedback enregistré et alerte retirée de la file."}