import streamlit as st
from web.helper.translator import t


def home_card():
    """Display preview of the home page"""
    st.page_link("web/pages/home.py", label=t("Home"), icon="🏠")
    st.markdown(t("""
        ### 🌌 Welcome!
        - Explore exoplanets from NASA missions
        - Interactive 3D planet simulations
        - Advanced analysis tools
    """))


def exoplanet_predictor_card():
    """Display preview of the exoplanet predictor"""
    st.page_link("web/pages/Exoplanet_Predictor.py", label=t("Exoplanet Predictor"), icon="🌌")
    st.markdown(t("""
        ### 🔭 Planet Prediction
        - Manual input or CSV upload
        - K2, KOI, TESS data support
        - Interactive 3D previews
    """))
    
def exoplanet_flux_predictor_card():
    """Display preview of the exoplanet predictor"""
    st.page_link("web/pages/Exoplanet_Flux_Prediction.py", label=t("Exoplanet Predictor"), icon="💫")
    st.markdown(t("""
        ### 🔭 Planet Prediction
        - Cached AI model for fast predictions
        - FITS, CSV, and ASCII data support
        - Secure user and database integration
    """))


def history_card():
    """Display preview of prediction history"""
    st.page_link("web/pages/history.py", label=t("History"), icon="📊")
    st.markdown(t("""
        ### 📊 Prediction History
        - View past predictions
        - Download results
        - Filter by type and date
    """))


def models_docs_card():
    """Display preview of model documentation"""
    st.page_link("web/pages/docs.py", label=t("Models Docs"), icon="📄")
    st.markdown(t("""
        ### 📄 Model Documentation
        - Technical details
        - Usage guidelines
        - Interactive chat support
    """))


def help_card():
    """Display preview of help section"""
    st.page_link("web/pages/helps.py", label=t("Help"), icon="❓")
    st.markdown(t("""
        ### ❓ Help & Support
        - Usage instructions
        - Troubleshooting
        - Contact support
    """))


def get_all_cards():
    """Return mapping of all card functions"""
    return {
        t("Home"): home_card,
        t("Exoplanet Predictor"): exoplanet_predictor_card,
        t("Exoplanet Flux Predictor") : exoplanet_flux_predictor_card,
        t("History"): history_card,
        t("Models Docs"): models_docs_card,
        t("Help"): help_card
    }
