import streamlit as st
import pandas as pd # au cas où
from utils.ui_style import load_css, create_header, create_sidebar, create_footer, setup_page_config, \
    show_home_page_content, apply_button_style
from utils.auth import check_authentication, logout_button # 🌟 Ajoute logout_button ici

# --- 1. CONFIGURATION GLOBALE ---
#setup_page_config()

# --- 2. AUTHENTIFICATION ET STYLES ---
# 🌟 Désactive l'affichage du bouton automatique en haut
check_authentication(show_logout_now=False)

# Charge le CSS
load_css()
apply_button_style()

# En-tête de l'application
create_header()

# Sidebar de navigation (génère tes boutons et ton texte "À propos")
selected_page = create_sidebar()

# 🌟 CODE CSS & BOUTON : Injecté directement dans app.py pour toutes les pages
st.sidebar.markdown("""
    <style>
    /* Force le conteneur complet de la sidebar à utiliser Flexbox */
    [data-testid="stSidebarUserContent"] {
        display: flex;
        flex-direction: column;
        min-height: 82vh;
    }
    /* Pousse le dernier élément (notre bouton) tout en bas de l'écran */
    [data-testid="stSidebarUserContent"] > div:last-child {
        margin-top: auto;
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Affiche le bouton proprement tout en bas de la sidebar
logout_button(key="logout_main_app")


# --- 3. LOGIQUE DE NAVIGATION DES PAGES ---
# Si on est sur l'accueil (aucune page cliquée ou page d'accueil par défaut)
if selected_page == "🏠 Accueil" or not selected_page: # Ajuste selon le nom dans ton create_sidebar()
    show_home_page_content()

elif selected_page == "📊 Dashboard":
    from pages import Dashboard
    Dashboard.show()

elif selected_page == "🔍 Détection ":
    from pages import Détection
    Détection.show()

elif selected_page == "🚨 Alertes en Temps Réel":
    from pages import Alertes
    Alertes.show()

# Pied de page
create_footer()