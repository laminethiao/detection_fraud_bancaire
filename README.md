# 💳 Système de Détection de Fraude Bancaire en Temps Réel

## 🌟 Vue d'ensemble du Projet

Ce projet implémente une solution de Machine Learning complète (MLOps) pour la détection en temps réel des transactions frauduleuses. L'objectif est de minimiser les pertes financières en fournissant aux analystes un outil rapide, interprétable et doté d'une boucle de rétroaction (Feedback Loop) pour l'amélioration continue du modèle.

### Technologies Clés

| Catégorie | Outils | Rôle dans le Projet |
| :--- | :--- | :--- |
| **Front-end** | **Streamlit** | Interface utilisateur interactive et tableaux de bord opérationnels. |
| **Back-end** | **FastAPI** | Serveur API REST ultra-rapide pour encapsuler et servir le modèle ML. |
| **Modèle** | **XGBoost** | Algorithme puissant et précis pour la classification binaire des transactions. |
| **Déploiement** | **GitHub, Streamlit Cloud** | Gestion du code source et déploiement continu du Front-end. |

## 📐 Architecture du Système

Le système est conçu autour d'une architecture moderne de microservices :

1.  **Streamlit Application (Front-end) :** Gère l'authentification, le Dashboard, et les pages interactives d'Alertes et de Détection.
2.  **FastAPI (Back-end) :** Expose deux points d'API critiques :
    * `/predict`: Reçoit les données de transaction et renvoie la prédiction de fraude en temps réel.
    * `/feedback`: Enregistre la validation manuelle de l'analyste (rétroaction) pour le réapprentissage futur du modèle.



## ✨ Fonctionnalités Principales

### 1. Détection en Temps Réel
Permet à l'utilisateur de simuler une nouvelle transaction. L'application envoie les 28 variables PCA au service FastAPI, et affiche instantanément la probabilité de fraude fournie par le modèle XGBoost.

### 2. Triage des Alertes & Interprétabilité
La page **Alertes** présente les transactions suspectes.
* **Visualisation PCA :** Utilisation de la Réduction de Composantes Principales (PCA) pour projeter la transaction suspecte par rapport à l'ensemble du dataset, permettant à l'analyste de **visualiser l'anomalie**.
* **Action Analytique :** L'analyste peut **Confirmer Fraude** ou **Confirmer Normal**, enregistrant ainsi des données de rétroaction de haute qualité.

### 3. Tableau de Bord Opérationnel
Un Dashboard dynamique pour le suivi des performances opérationnelles :
* **Performance du Modèle :** Rappel (Recall), Précision, Taux de Fausses Alertes.
* **Données de Rétroaction :** Suivi en temps réel du nombre de fraudes et de transactions normales confirmées manuellement (la boucle de feedback).

## 🚀 Mise en Route

### Prérequis

* Python 3.9+
* Git

### Installation et Lancement

1.  **Clonez le dépôt :**
    ```bash
    git clone [Votre URL GitHub]
    cd detection_fraud_bancaire
    ```

2.  **Créez et activez l'environnement virtuel :**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Linux/Mac
    # ou venv\Scripts\activate.bat sur Windows
    ```

3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Lancez l'API FastAPI (Back-end) :**
    ```bash
    uvicorn api.main:app --reload
    ```
    (Assurez-vous d'être dans le bon répertoire pour `api.main:app`)

5.  **Lancez l'application Streamlit (Front-end) :**
    ```bash
    streamlit run auth.py
    ```

L'application sera accessible dans votre navigateur à l'adresse `http://localhost:8501`. Utilisez `user@gmail.com` / `thiao123` pour vous connecter.
