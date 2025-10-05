import streamlit as st
import pandas as pd
import numpy as np
if "init" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(
        np.random.randn(20, 3), columns=["a", "b", "c"]
    )
    st.session_state.map_data = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=["lat", "lon"],
    )
    st.session_state.init = True


pages = [
    st.Page("web/pages/home.py", title="Home", icon="🏠"),
    st.Page("web/pages/Exoplanet_Predictor.py", title="Exoplanet Predictor", icon="🌌"),
    st.Page("web/pages/text.py", title="Text", icon="🌌"),
    st.Page("web/pages/data.py", title="Data", icon="🌌"),
    #st.Page("web/pages/charts.py", title="Charts", icon="🌌"),
    st.Page("web/pages/media.py", title="Media", icon="🌌"),
    st.Page("web/pages/layouts.py", title="Layouts", icon="🌌"),
    st.Page("web/pages/chat.py", title="Models Docs", icon="📄"),
    st.Page("web/pages/status.py", title="Helps", icon="❓"),
]

page = st.navigation(pages)
page.run()

with st.sidebar.container(border=True):
    st.write("Hello!, Ha Tri Kien")
    st.button("Logout")
    
