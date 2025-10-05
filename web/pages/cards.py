import streamlit as st

def home_card():
    """Display preview of the home page"""
    st.page_link("web/pages/home.py", label="Home", icon="🏠")
    st.markdown("""
        ### 🌌 Welcome!
        - Explore exoplanets from NASA missions
        - Interactive 3D planet simulations
        - Advanced analysis tools
    """)

def exoplanet_predictor_card():
    """Display preview of the exoplanet predictor"""
    st.page_link("web/pages/Exoplanet_Predictor.py", label="Exoplanet Predictor", icon="🌌")
    st.markdown("""
        ### 🔭 Planet Prediction
        - Manual input or CSV upload
        - K2, KOI, TESS data support
        - Interactive 3D previews
    """)

def history_card():
    """Display preview of prediction history"""
    st.page_link("web/pages/history.py", label="History", icon="📊")
    st.markdown("""
        ### � Prediction History
        - View past predictions
        - Download results
        - Filter by type and date
    """)

def models_docs_card():
    """Display preview of model documentation"""
    st.page_link("web/pages/chat.py", label="Models Docs", icon="📄")
    st.markdown("""
        ### 📄 Model Documentation
        - Technical details
        - Usage guidelines
        - Interactive chat support
    """)

def help_card():
    """Display preview of help section"""
    st.page_link("web/pages/status.py", label="Help", icon="❓")
    st.markdown("""
        ### ❓ Help & Support
        - Usage instructions
        - Troubleshooting
        - Contact support
    """)

# Function to get all cards for easier management
def get_all_cards():
    return {
        "Home": home_card,
        "Exoplanet Predictor": exoplanet_predictor_card,
        "History": history_card,
        "Models Docs": models_docs_card,
        "Help": help_card
    }