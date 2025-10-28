import streamlit as st

import pandas as pd
import plotly.graph_objects as go


def load_css():
    """Charge le CSS personnalisé pour l'application"""
    st.markdown("""
    <style>
    /* Style général */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #262730;
    }

    /* Titres */
    h1, h2, h3 {
        color: #00C8C8 !important;
        font-weight: 700;
    }

    /* Cartes et conteneurs */
    .card {
        background-color: #262730;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00C8C8;
        margin-bottom: 20px;
    }

    /* Alertes fraude */
    .fraud-alert {
        background-color: #FF4B4B;
        color: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #FF0000;
        font-weight: bold;
    }

    .no-fraud {
        background-color: #00C8C8;
        color: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #009999;
        font-weight: bold;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #00C8C8 !important;
    }

    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #00C8C8;
    }

    .stNumberInput>div>div>input {
        background-color: #262730;
        color: white;
        border: 1px solid #00C8C8;
    }

    /* Séparateurs */
    .hr-gradient {
        height: 2px;
        background: linear-gradient(90deg, #00C8C8, #FF4B4B);
        margin: 2rem 0;
        border: none;
    }
    /* STYLISATION DES PAGES DE NAVIGATION */
    [data-testid="stSidebarNav"] > ul > li > div > a,
    [data-testid="stSidebarNav"] > ul > li > div > span > a {
        background-color: #00C8C8 !important;
        color: white !important;
        border-radius: 4px !important;
        margin: 3px 0 !important;                      /* ESPACE ENTRE LES BARRES */
        padding: 3px 8px 3px 6px !important;           /* PLUS DE PADDING À GAUCHE */
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid #009999 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;        /* ALIGNEMENT À GAUCHE */
        font-size: 12px !important;
        min-height: 26px !important;                   /* HAUTEUR RÉDUITE */
        text-align: left !important;                   /* TEXTE À GAUCHE */
        width: 85% !important;                         /* LARGEUR ENCORE RÉDUITE */
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* SURVOL DES PAGES */
    [data-testid="stSidebarNav"] > ul > li > div > a:hover,
    [data-testid="stSidebarNav"] > ul > li > div > span > a:hover {
        background-color: #10B981 !important;
        color: white !important;
        transform: scale(1.02) !important;
        border: 1px solid #059669 !important;
    }

    /* PAGE ACTIVE */
    [data-testid="stSidebarNav"] > ul > li > div > a[aria-current="page"],
    [data-testid="st-SidebarNav"] > ul > li > div > span > a[aria-current="page"] {
        background-color: #009999 !important;
        color: white !important;
        border: 1px solid #10B981 !important;
        font-weight: 600 !important;
    }

    /* ICÔNES DES PAGES - ALIGNEMENT GAUCHE PARFAIT */
    [data-testid="stSidebarNav"] > ul > li > div > a::before,
    [data-testid="stSidebarNav"] > ul > li > div > span > a::before {
        content: "•";
        margin-right: 5px;
        color: white;
        font-weight: bold;
        font-size: 10px;
        line-height: 1;
        flex-shrink: 0;                               /* EMPÊCHE LE RÉTRÉCISSEMENT */
    }

    [data-testid="stSidebarNav"] > ul > li > div > a:hover::before,
    [data-testid="stSidebarNav"] > ul > li > div > span > a:hover::before {
        color: white;
    }

    [data-testid="stSidebarNav"] > ul > li > div > a[aria-current="page"]::before,
    [data-testid="stSidebarNav"] > ul > li > div > span > a[aria-current="page"]::before {
        content: "➤";
        color: white;
        font-size: 9px;
        margin-right: 4px;
    }


    """, unsafe_allow_html=True)


def setup_page_config():
    """Configure la page Streamlit"""
    st.set_page_config(
        page_title="Système de Détection de Fraude Bancaire",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def create_header():
    """Crée l'en-tête de l'application"""
    col1, col2 = st.columns([1, 6])
    with col1:
        # Utilisez du HTML pour l'emoji
        st.markdown("<h1>💳</h1>", unsafe_allow_html=True)
    with col2:
        st.title("Système de Détection de Fraude Bancaire")
        st.caption("Détection en temps réel des transactions frauduleuses")

# info de application
def show_home_page_content():
    """Affiche le contenu principal de la page d'accueil (explications, modèle, métriques)."""

    st.title("🏠 Bienvenue sur l'Application de Détection de Fraude")

    st.markdown("""
        Cette application est une solution avancée pour détecter et analyser en temps réel les transactions financières potentiellement frauduleuses. 
        Elle combine un puissant modèle de **Machine Learning** avec une interface utilisateur intuitive pour aider les analystes à trier et à valider rapidement les alertes.
    """)
    # NOTE: L'image est une illustration temporaire
    st.image("https://placehold.co/1200x600/E5E5E5/000000?text=Illustration+du+tableau+de+bord", use_container_width=True)

    st.markdown("---")

    st.header("🎯 Objectif de l'Application")
    st.markdown("""
        L'objectif principal est de **réduire le temps nécessaire** à l'identification de la fraude en signalant uniquement les transactions les plus suspectes. Cela permet aux analystes de se concentrer sur les cas à fort risque, améliorant ainsi l'efficacité et la sécurité du système financier.

        Les fonctionnalités clés incluent :
        - **Détection en temps réel** : Un modèle prédit la probabilité de fraude pour chaque nouvelle transaction.
        - **File d'attente d'alertes** : Les transactions suspectes sont placées dans une file d'attente pour être triées.
        - **Visualisation de l'anomalie** : Chaque alerte est accompagnée d'un graphique qui montre pourquoi elle est considérée comme anormale.
        - **Rétroaction du modèle** : Les analystes peuvent valider ou rejeter les alertes, ce qui permet d'améliorer la performance du modèle au fil du temps.
    """)

    st.markdown(
        """
        Pour la détection des anomalies, nous utilisons l'algorithme de pointe **XGBoost 
        (Extreme Gradient Boosting)**. Ce modèle est particulièrement bien adapté à ce type de problème car :

        1.  **Modèle Supervisé de Référence :** Entraîné sur des données historiques labellisées, 
            XGBoost offre une précision de classification supérieure pour capturer la fraude 
            avec un excellent taux de rappel (Recall).
        2.  **Gestion du Déséquilibre :** Il gère efficacement le fort déséquilibre des classes 
            (très peu de fraudes) typique des données bancaires.
        3.  **Performance en Temps Réel :** Optimisé pour la vitesse, il permet des inférences 
            rapides, cruciales pour la détection temps réel via l'API FastAPI.

        ### Note Importante

        Le modèle est chargé à l'intérieur de l'API FastAPI (`/api/main.py`) et est encapsulé 
        avec le Scaler pour garantir que les nouvelles transactions sont traitées exactement 
        comme les données d'entraînement.
        """)

    st.markdown("---")

    st.header("📈 Performance du Modèle")
    st.markdown("""
        La performance du modèle a été évaluée sur des données historiques. Voici un aperçu des métriques clés (valeurs simulées pour démonstration) :
    """)

    # Tableau des métriques de performance
    metrics_data = {
        "Métrique": ["Précision", "Rappel (Recall)", "F1-Score", "AUC-ROC"],
        "Valeur": ["95.5%", "88.2%", "91.7%", "0.94"]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df.set_index('Métrique'))

    st.markdown("""
        - **Précision** : Indique le pourcentage de fraudes détectées qui sont de véritables fraudes.
        - **Rappel** : Mesure le pourcentage de toutes les fraudes réelles que notre modèle a réussi à détecter. C'est une métrique **critique**.
    """)

    st.subheader("Matrice de Confusion")

    # Création de la matrice de confusion simulée avec Plotly
    labels = ["Normal", "Fraude"]
    z_matrix = [[98500, 150], [180, 520]] # Vrais Négatifs, Faux Positifs, Faux Négatifs, Vrais Positifs

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=labels,
        y=labels,
        colorscale='Viridis'
    ))

    fig.update_layout(
        title_text='Matrice de Confusion (Données Simulées)',
        xaxis_title_text="Prédiction",
        yaxis_title_text="Valeur Réelle",
        xaxis={'side': 'bottom'},
        height=450,
        width=450
    )
    st.plotly_chart(fig)

    st.markdown("""
        * **Vrais Positifs (VP)** : Le modèle a prédit une fraude, et c'est une vraie fraude. (520)
        * **Faux Négatifs (FN)** : Le modèle a prédit une transaction normale, mais c'est une fraude réelle (180). L'objectif est de minimiser ce nombre.
    """)


def create_sidebar():
    """Crée seulement la section À propos dans la sidebar"""
    with st.sidebar:
        st.info("""
        *À propos:*
        Système de détection de fraude utilisant le Machine Learning
        pour analyser les transactions bancaires en temps réel.

        """)


def create_footer():
    """Crée le pied de page"""
    st.markdown("---")
    st.caption("""
    *Système de Détection de Fraude Bancaire* • 
    Développé avec XGBoost et Streamlit • 
    © 2025 - Tous droits réservés
    """)


def apply_button_style():
    """Applique un style cohérent aux boutons"""
    st.markdown("""
        <style>
        /* ciblons spécifiquement les boutons de chargement et d'analyse */
        div.stButton button, div.stFormSubmitButton button {
            background-color: #00C8C8;
            color: white;
            border-radius: 8px;
            border: 1px solid #009999;
            padding: 10px 24px;
            font-weight: 600;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease-in-out;
        }
        div.stButton button:hover, div.stFormSubmitButton button:hover {
            background-color: #009999;
            border-color: #006666;
            transform: translateY(-2px);
        }
        </style>
    """, unsafe_allow_html=True)


def apply_login_form_style():
    """Applique des styles spécifiques pour le formulaire de connexion."""
    st.markdown(
        """
        <style>
        .stForm {
            background-color: rgba(0, 0, 0, 0.6);
            padding: 30px;
            border-radius: 10px;
            border: 2px solid #00C8C8;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
        }
        .stForm .stButton > button {
            width: 100%;
            background-color: #00C8C8;
            color: white;
            border: none;
            padding: 10px;
            border-radius: 5px;
        }
        .stForm .stTextInput > div > div > input {
            background-color: #262730;
            color: white;
            border: 1px solid #00C8C8;
            border-radius: 5px;
        }
        .stForm .stTextInput > label {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Dans utils/ui_style.py

def set_background():
    """Définit l'image de fond et les styles globaux pour l'application."""
    st.markdown(
        f"""
            <style>

            /* SÉLECTEUR RACINE AGRESSIF pour forcer le fond sur toute la page avant st.stop() */
            #root > div:nth-child(1) {{
                /* Si l'image n'est pas chargée, au moins le fond est sombre */
                background-color: #0E1117; 
            }}

            /* Cible l'application entière pour l'image de fond */
            .stApp {{
                background-image: url("https://images.unsplash.com/photo-1542281242-d621b1b0f19c?q=80&w=2938&auto=format&fit=crop");
                background-size: cover;
                background-position: center;
                background-attachment: fixed; /* Rendre l'image fixe */
            }}

            /* S'assurer que le contenu du conteneur principal est transparent pour voir l'image */
            .main {{
                background-color: transparent !important; 
            }}

            /* S'assurer que les titres sont lisibles sur l'image */
            .stMarkdown, .stTitle, .stHeader, .stSubheader {{
                color: white;
                text-shadow: 2px 2px 4px #000000;
            }}

            </style>
            """,
        unsafe_allow_html=True
    )

def custom_sidebar_style():
    """Applique un style CSS personnalisé à la barre latérale."""
    st.markdown(
        """
        <style>
        .css-1d391gc {{
            background-color: #333333;
            color: white;
        }}
        .css-1d391gc .st-br {{
            color: #ffffff;
        }}
        </style>
        """,
        unsafe_allow_html=True)

#style de connexion
def apply_auth_button_style():
    """
    Applique le style CSS spécifique aux boutons de connexion et d'inscription
    en leur donnant la même couleur que la navigation.
    """
    st.markdown("""
    <style>
    /* Cibler les boutons du formulaire de connexion */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #00C8C8 !important;
        border-color: #00C8C8 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease-in-out !important;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #009999 !important;
        border-color: #006666 !important;
        transform: translateY(-2px) !important;
    }

    /* S'assurer que les deux boutons ont le même style */
    div[data-testid="stFormSubmitButton"]:nth-of-type(2) button {
        background-color: #00C8C8 !important;
        border-color: #00C8C8 !important;
        color: white !important;
    }

    div[data-testid="stFormSubmitButton"]:nth-of-type(2) button:hover {
        background-color: #009999 !important;
        border-color: #006666 !important;
    }
    </style>
    """, unsafe_allow_html=True)