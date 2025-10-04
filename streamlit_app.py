import time
import pandas as pd
import numpy as np
from typing import Optional

import streamlit as st
from streamlit.navigation.page import StreamlitPage

from web.pages.cards import (
    widgets_card,
    text_card,
    dataframe_card,
    charts_card,
    media_card,
    layouts_card,
    chat_card,
    status_card,
)
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
    if "init" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame(
            np.random.randn(20, 3), 
            columns=["a", "b", "c"]
        )
        st.session_state.map_data = pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=["lat", "lon"]
        )
        st.session_state.init = True

# Define application pages
pages = [
    st.Page("web/pages/home.py", title="Home", icon="🏠"),
    st.Page("web/pages/widgets.py", title="Widgets", icon="🎛️"),
    st.Page("web/pages/text.py", title="Text", icon="📝"),
    st.Page("web/pages/data.py", title="Data", icon="📊"),
    st.Page("web/pages/charts.py", title="Charts", icon="📈"),
    st.Page("web/pages/media.py", title="Media", icon="🖼️"),
    st.Page("web/pages/layouts.py", title="Layouts", icon="📐"),
    st.Page("web/pages/chat.py", title="Chat", icon="💬"),
    st.Page("web/pages/status.py", title="Status", icon="ℹ️"),
]

def render_sidebar_header(auth_user: str, authorizer: AuthHub, page: StreamlitPage) -> None:
    """Render the sidebar header with user information and navigation."""
    with st.sidebar:
        st.markdown(f"👋 Welcome, **{auth_user}**")
        if st.button("Logout", type="secondary"):
            authorizer.logout()
        st.divider()

def render_sidebar_content(page: StreamlitPage) -> None:
    """Render the sidebar content based on current page."""
    with st.sidebar.container(height=310):
        card_mapping = {
            "Widgets": widgets_card,
            "Text": text_card,
            "Data": dataframe_card,
            "Charts": charts_card,
            "Media": media_card,
            "Layouts": layouts_card,
            "Chat": chat_card,
            "Status": status_card
        }
        
        if page.title in card_mapping:
            card_mapping[page.title]()
        else:
            st.page_link("web/pages/home.py", 
                        label="Home", 
                        icon="🏠")
            st.markdown("""
                ### Welcome! 
                Select a page from above to explore different features.
                This sidebar provides quick previews of each section.
            """)

def dashboard(authorizer: AuthHub) -> None:
    """Main dashboard rendering function."""
    initialize_session_state()
    
    if "auth_user" not in st.session_state:
        return

    page = st.navigation(pages)
    page.run()
    
    render_sidebar_header(st.session_state["auth_user"], authorizer, page)

    render_sidebar_content(page)

def main() -> None:
    """Application entry point with authentication."""
    authorizer = AuthHub(connect_db())
    
    if authorizer.try_authorize_by_cookie():
        dashboard(authorizer)
    else:
        from web.pages import login
        login.main(authorizer)

if __name__ == "__main__":
    main()