import streamlit as st
from ..utils.authorizer import AuthHub
from ..utils.cookie import redirect


def main(authorizer: AuthHub):
    st.header("🔐 Please log in")
    with st.form(key="login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if authorizer.login(username, password):
            st.query_params.pop("page", None)
            authorizer.wait_for_cookie(0.2)
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    col1, col2, col3 = st.columns([0.12, 0.1, 0.78])
    with col1:
        st.markdown(
            """
            <div style='line-height: 2.5;'>
                New to our platform?
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col2:
        if st.button("Sign Up", type="primary"):
            redirect("signup")
    with col3:
        # Empty column for spacing
        pass