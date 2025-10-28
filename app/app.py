import streamlit as st
# Importez uniquement les fonctions nécessaires
from utils.ui_style import load_css, create_header, create_sidebar, create_footer, setup_page_config, \
    show_home_page_content
from utils.auth import check_authentication

# --- 1. CONFIGURATION GLOBALE (DOIT ÊTRE LE PREMIER APPEL STREAMLIT) ---
#setup_page_config()

# --- 2. AUTHENTIFICATION ET STYLES ---
# ATTENTION : Si vous n'êtes pas sur la page de connexion, check_authentication()
# va rediriger immédiatement via st.switch_page().
check_authentication()

# Charge le CSS
load_css()

# En-tête de l'application
create_header()
show_home_page_content()
# Sidebar de navigation
selected_page = create_sidebar()

# --- 3. LOGIQUE DE NAVIGATION DES PAGES ---

# La logique de navigation doit faire référence aux modules Python de vos pages.
# On utilise la sémantique de l'API Multi-Page de Streamlit ici pour inclure le contenu.

if selected_page == "📊 Dashboard":
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

# ← ICI C'EST MAINTENANT UNE LIGNE VIDE (vous l'avez ajoutée)