import streamlit as st
from ..utils.authorizer import AuthHub



def main(authorizer: AuthHub):
    st.header("🔐 Please log in")
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if authorizer.login(username, password):
            st.rerun()
        else:
            st.error("❌ Invalid username or password")