import streamlit as st
from ..utils.authorizer import AuthHub, validate_password
from ..utils.routing import redirect


def main(authorizer: AuthHub):
    st.header("🚀 Create Your Account")
    
    # Show password requirements
    with st.expander("Password Requirements", expanded=True):
        st.markdown("""
        Your password must have:
        - At least 8 characters
        - At least one uppercase letter (A-Z)
        - At least one lowercase letter (a-z)
        - At least one number (0-9)
        - At least one special character (!@#$%^&*(),.?":{}|<>)
        """)
    
    with st.form(key="signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Sign Up")

    if submitted:
        if not username or not password:
            st.error("❌ Please fill in all fields")
            return
            
        if password != confirm_password:
            st.error("❌ Passwords do not match")
            return
            
        is_valid, message = validate_password(password)
        if not is_valid:
            st.error(f"❌ {message}")
            return
            
        if authorizer.signup(username, password):
            st.query_params.pop("page", None)
            authorizer.wait_for_cookie(0.2)
            st.rerun()
        else:
            st.error("❌ Username already existed")

    col1, col2, col3 = st.columns([0.15, 0.1, 0.75])
    with col1:
        st.markdown(
            """
            <div style='line-height: 2.5;'>
                Already have an account?
            </div>
            """, 
            unsafe_allow_html=True
        )
    with col2:
        if st.button("Log In", type="primary"):
            redirect("login")
    with col3:
        # Empty column for spacing
        pass