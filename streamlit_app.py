import time
import pandas as pd
import numpy as np
from typing import Optional

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from web.utils.authorizer import AuthHub
from web.db import connect_db
from web.utils.cookie import redirect


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

pages = [
    st.Page("web/pages/home.py", title="Home", icon="🏠"),
    st.Page("web/pages/Exoplanet_Predictor.py", title="Exoplanet Predictor", icon="🌌"),
    st.Page("web/pages/Exoplanet_Flux_Prediction.py", title="Exoplanet Flux Prediction", icon="💫"),
    st.Page("web/pages/history.py", title="History", icon="📊"),
    # st.Page("web/pages/data.py", title="Data", icon="🌌"),
    # st.Page("web/pages/media.py", title="Media", icon="🌌"),
    # st.Page("web/pages/layouts.py", title="Layouts", icon="🌌"),
    st.Page("web/pages/chat.py", title="Models Docs", icon="📄"),
    st.Page("web/pages/status.py", title="Helps", icon="❓"),
]

def render_sidebar_header(auth_user: str, authorizer: AuthHub) -> None:
    """Render the sidebar header with user information and navigation."""
    with st.sidebar:
        st.markdown(f"👋 Welcome, **{auth_user}**")
        if st.button("Logout", type="secondary"):
            authorizer.logout()
            st.session_state.pop("init", None)
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
    
    if "auth_user" not in st.session_state:
        return
    
    initialize_session_state()

    page = st.navigation(pages)
    page.run()
    
    render_sidebar_header(st.session_state["auth_user"], authorizer)

    render_sidebar_content(page)

def main() -> None:
    """Application entry point with authentication."""
    authorizer = AuthHub(connect_db())
    
    if authorizer.try_authorize_by_cookie():
        dashboard(authorizer)
    else:
        from web.pages import login, signup
        (signup if st.query_params.get("page") == "signup" else login).main(authorizer)

if __name__ == "__main__":
    main()