import streamlit as st
import time
import re

# Corrected import for ui_style
from utils.ui_style import apply_login_form_style, set_background, custom_sidebar_style

from utils.ui_style import apply_auth_button_style

# --- Système de connexion simplifié (pour démonstration) ---
# Vous pouvez connecter ce dictionnaire à une base de données dans une version plus avancée
USERS = {
    "user@gmail.com": "thiao123",
    "admin@example.com": "adminpass"
}

# Appliquer les styles au chargement du module



def validate_email(email):
    """Valide le format de l'email en utilisant une expression régulière."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)


def validate_password(password):
    """
    Valide le mot de passe :
    - Au moins 8 caractères.
    - Contient au moins une lettre et un chiffre.
    """
    if len(password) < 8:
        return "Le mot de passe doit contenir au moins 8 caractères."
    if not re.search(r"[a-zA-Z]", password):
        return "Le mot de passe doit contenir au moins une lettre."
    if not re.search(r"\d", password):
        return "Le mot de passe doit contenir au moins un chiffre."
    return None


def login_user(username, password):
    """Vérifie les identifiants de l'utilisateur et met à jour l'état de la session."""
    if not validate_email(username):
        st.error("Format d'email invalide.")
        return

    if username in USERS and USERS[username] == password:
        st.session_state["logged_in"] = True
        st.session_state["user_email"] = username
        st.success(f"Bienvenue, {username} !")
        time.sleep(1)
        st.rerun()
    else:
        st.error("Nom d'utilisateur ou mot de passe incorrect.")


def logout_user():
    """Déconnecte l'utilisateur en réinitialisant les variables de session."""
    st.session_state["logged_in"] = False
    st.session_state["user_email"] = None
    st.rerun()


def register_user(username, password, confirm_password):
    """Simule l'inscription d'un nouvel utilisateur avec une validation améliorée."""
    # Validation de l'email
    if not validate_email(username):
        st.error("Format d'email invalide.")
        return

    # Validation de l'existence de l'utilisateur
    if username in USERS:
        st.error("Ce nom d'utilisateur existe déjà.")
        return

    # Validation du mot de passe
    password_error = validate_password(password)
    if password_error:
        st.error(password_error)
        return

    if password != confirm_password:
        st.error("Les mots de passe ne correspondent pas.")
        return

    # Si toutes les validations passent
    USERS[username] = password
    st.session_state["logged_in"] = False
    st.session_state["user_email"] = None
    st.success("Inscription réussie ! Vous pouvez maintenant vous connecter.")
    time.sleep(2)
    st.rerun()


def show_login_form():
    """Affiche le formulaire de connexion et d'inscription."""

    set_background()
    custom_sidebar_style()
    apply_login_form_style()
    apply_auth_button_style()

    apply_login_form_style()

    # Centrer le contenu en utilisant des colonnes vides
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔑 Connexion")
        st.markdown("Veuillez vous connecter pour accéder à l'application.")

        # Formulaire de connexion
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur (Email)")
            password = st.text_input("Mot de passe", type="password")
            login_button = st.form_submit_button("Se connecter", type="primary")
            if login_button:
                login_user(username, password)

        # Formulaire d'inscription
        st.subheader("Nouveau ici ? Créez un compte")
        with st.form("register_form"):
            new_username = st.text_input("Nouveau nom d'utilisateur (Email)")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmer le mot de passe", type="password")
            register_button = st.form_submit_button("S'inscrire", type="primary")
            if register_button:
                register_user(new_username, new_password, confirm_password)


def logout_button(key="logout_default"):
    """Affiche un bouton de déconnexion dans la barre latérale avec une clé unique."""
    st.sidebar.button("Se déconnecter", on_click=logout_user, type="secondary", key=key)



def is_logged_in():
    """Vérifie l'état de connexion de l'utilisateur."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    return st.session_state["logged_in"]


def check_authentication(show_logout_now=True):
    """Vérifie l'état de l'authentification et gère l'affichage du formulaire ou de la page principale."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        show_login_form()
        st.stop()
    else:
        st.sidebar.header(f"Bonjour, {st.session_state.get('user_email', 'Utilisateur')} 👋")
        # 🌟 On affiche le bouton uniquement si demandé
        if show_logout_now:
            logout_button()


if __name__ == '__main__':
    # Exemple d'utilisation de la fonction de vérification
    check_authentication()
