import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import numpy as np

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
    Class: int

# Initialisation de l'API FastAPI
app = FastAPI()

# Variables globales pour le modèle et le scaler
model = None
scaler = None
# Fichier de stockage pour la rétroaction
FEEDBACK_FILE = "feedback_data.csv"

# Fonction pour charger les fichiers au démarrage de l'application
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
@app.get("/")
async def home():
    """
    Affiche un message de bienvenue pour confirmer que l'API est en ligne.
    """
    return {
        "message": "API de détection de fraude en cours d'exécution."
    }

# Endpoint de prédiction
@app.post("/predict")
async def predict_transaction(transaction: Transaction):
    """
    Prédit si une transaction est frauduleuse (1) ou non (0).
    """
    global model, scaler

    # Vérifie si le modèle et le scaler ont été chargés
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Le service est indisponible.")

    try:
        # Convertir les données de l'API en DataFrame
        df = pd.DataFrame([transaction.model_dump()])

        # Normaliser les variables 'Time' et 'Amount'
        scaled_features = ['Time', 'Amount']
        df[scaled_features] = scaler.transform(df[scaled_features])

        # Faire la prédiction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0][1]

        # Retourner la prédiction et la probabilité
        return {
            "prediction": int(prediction),
            "probability": float(prediction_proba),
            "confidence": "Haute" if prediction_proba > 0.8 else ("Moyenne" if prediction_proba > 0.5 else "Basse")
        }
    except Exception as e:
        # Retourne une erreur détaillée en cas de problème
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction: {e}")

# Endpoint pour la rétroaction
@app.post("/feedback")
def submit_feedback(data: FeedbackData):
    """
    Enregistre la rétroaction manuelle d'un analyste pour le réentraînement futur.
    """
    try:
        df = pd.DataFrame([data.model_dump()])
        # Écrire dans le fichier CSV. Si le fichier n'existe pas, créer l'en-tête.
        header = not os.path.exists(FEEDBACK_FILE)
        df.to_csv(FEEDBACK_FILE, mode='a', header=header, index=False)
        print(f"✅ Rétroaction enregistrée : {data.model_dump()}")
        return {"message": "Rétroaction enregistrée avec succès", "received_data": data.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Échec de l'enregistrement de la rétroaction : {e}")

# Endpoint pour l'historique des alertes - LISANT DU FICHIER CSV
@app.get("/alerts")
def get_alerts():
    """
    Récupère la liste de toutes les transactions de rétroaction à partir du fichier CSV.
    """
    if os.path.exists(FEEDBACK_FILE):
        try:
            # Lire le fichier CSV et le convertir en liste de dictionnaires
            df_feedback = pd.read_csv(FEEDBACK_FILE)
            return {"alerts": df_feedback.to_dict('records')}
        except pd.errors.EmptyDataError:
            # Gérer le cas où le fichier est vide
            return {"alerts": []}
    return {"alerts": []}
