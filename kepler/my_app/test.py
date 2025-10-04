# login_demo_option_a.py
import streamlit as st
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

from utils import delete_client_cooke, CookieUtil

# -----------------------
# Configuration
# -----------------------
SECRET_KEY = "a3f9d2e7c1b6f8a0d5e9c4b2a7f1e3d6"  # replace in production
signer = TimestampSigner(SECRET_KEY)

USERS = {"alice": "wonderland", "bob": "builder", "carol": "s3cr3t"}
COOKIE_NAME = "streamlit_auth_token"
TOKEN_MAX_AGE = 60 * 60 * 24 * 7  # 1 week (seconds)

# -----------------------
# Helpers
# -----------------------
def _create_token(username: str) -> str:
    return signer.sign(username.encode()).decode()

def _verify_token(token: str) -> str | None:
    try:
        data = signer.unsign(token.encode(), max_age=TOKEN_MAX_AGE)
        return data.decode()
    except (BadSignature, SignatureExpired):
        return None

def login(username: str, password: str) -> bool:
    return USERS.get(username) == password

def _extract_token(raw):
    """
    Handle different return shapes from cookie_manager.get(...)
    - None
    - plain string
    - dict-like {'value': token}
    - object with .value
    """
    if not raw:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("value") or raw.get("cookie") or None
    return getattr(raw, "value", None)

# -----------------------
# Main app
# -----------------------
def main():
    st.set_page_config(page_title="Secure Demo Login", page_icon="🔐")
    cookie_manager = CookieUtil()

    auth_user = st.session_state.get("auth_user")

    token = None
    if not auth_user:
        if st.session_state.get("logged_out"):
            # pop the flag and avoid trusting cookies in this run
            st.session_state.pop("logged_out", None)
            token = None
        else:
            raw = cookie_manager.get(COOKIE_NAME)
            token = _extract_token(raw)
            if token:
                verified = _verify_token(token)
                if verified:
                    st.session_state["auth_user"] = verified
                    auth_user = verified

    # ---------- Authenticated view ----------
    if auth_user:
        st.success(f"✅ Logged in as **{auth_user}**")

        # Logout button (do NOT rerun immediately)
        if st.button("Log out"):
            st.session_state.pop("auth_user", None)
            st.session_state["logged_out"] = True
            cookie_manager.delete(COOKIE_NAME)
            delete_client_cooke(COOKIE_NAME)

            return

        st.header("🚀 Protected Dashboard")
        st.write("Only visible after a successful login.")
        st.line_chart({"data": [1, 3, 2, 5, 4]})

    # ---------- Logged-out / login form ----------
    else:
        st.header("🔐 Please log in")
        with st.form(key="login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            if login(username, password):
                new_token = _create_token(username)
                cookie_manager.set(COOKIE_NAME, new_token)
                st.session_state["auth_user"] = username
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

if __name__ == "__main__":
    main()
