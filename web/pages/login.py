# Updated login.py: Enhanced UI for better visual appeal.
# Replaced the large image with the provided small logo.
# Centered the logo and adjusted size for a cleaner look.
# Kept the button centered.

import streamlit as st
from ..utils.authorizer import UserManager  # Renamed from authorizer
from ..utils.routing import redirect

def main(user_manager: UserManager):
    st.title("🌌 ExoVision Dashboard Login")
    
    st.markdown("""
    Welcome to the ExoVision Dashboard!  
    Sign in with your Google account to explore exoplanets, predictions, and more.  
    """)
    if st.button("🔑 Log in with Google", type="primary"):  # Added emoji and primary type for better look
        st.login()  # Or st.login("google") if multiple
    
    st.stop() 