import time
import pandas as pd
import numpy as np
from typing import Optional

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from web.utils.authorizer import AuthHub
from web.db import connect_db
from web.utils.routing import redirect


# Configure page settings
st.set_page_config(
    page_title="Kepler Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state() -> None:
    """Initialize session state with default values if not already present."""
    if "data_init" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame(
            np.random.randn(20, 3), 
            columns=["a", "b", "c"]
        )
        st.session_state.map_data = pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=["lat", "lon"]
        )
        st.session_state.data_init = True

# Define application pages

ALL_PAGES = [
    st.Page("web/pages/home.py", title="Home", icon="🏠"),
    st.Page("web/pages/Exoplanet_Predictor.py", title="Exoplanet Predictor", icon="🌌"),
    st.Page("web/pages/Exoplanet_Flux_Prediction.py", title="Exoplanet Flux Predictor", icon="💫"),
    st.Page("web/pages/history.py", title="History", icon="📊"),
    st.Page("web/pages/docs.py", title="Models Docs", icon="📄"),
    st.Page("web/pages/helps.py", title="Help", icon="❓"),
    st.Page("web/pages/about.py", title="About", icon="👤"),
]

GUEST_PAGES = [
    st.Page("web/pages/home.py", title="Home", icon="🏠"),
    st.Page("web/pages/helps.py", title="Help", icon="❓"),
    st.Page("web/pages/login.py", title="Login", icon="🔑"),
    st.Page("web/pages/signup.py", title="Sign Up", icon="✨"),
]

def render_sidebar_header(auth_user: str, authorizer: AuthHub) -> None:
    """Render the sidebar header with user information and navigation."""
    with st.sidebar:
        st.markdown(f"👋 Welcome, **{auth_user}**")
        if st.button("Logout", type="secondary"):
            st.session_state.pop("data_init", None)
            authorizer.logout()
            st.rerun()
        st.divider()

def render_sidebar_content(page: StreamlitPage) -> None:
    """Render the sidebar content based on current page."""
    from web.pages.cards import get_all_cards
    
    with st.sidebar.container(height=310):
        cards = get_all_cards()
        
        if page.title in cards:
            cards[page.title]()
        else:
            cards["Home"]()


def dashboard(authorizer: AuthHub) -> None:
    """Main dashboard rendering function."""
    initialize_session_state()
    page = st.navigation(ALL_PAGES)
    page.run()
    render_sidebar_header(st.session_state["auth_user"], authorizer)
    render_sidebar_content(page)


def main() -> None:
    """Application entry point with authentication."""
    authorizer = AuthHub(connect_db())
    if authorizer.try_authorize_by_cookie():
        dashboard(authorizer)
    else:
        page = st.navigation(GUEST_PAGES)
        if page.title in ["Login", "Sign Up"]:
            from web.pages import login, signup
            (signup if page.title == "Sign Up" else login).main(authorizer)
        else:
            page.run()

if __name__ == "__main__":
    main()