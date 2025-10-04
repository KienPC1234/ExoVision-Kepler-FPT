import time
import uuid
from passlib.context import CryptContext

import streamlit as st
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

from .cookie import CookieUtil, delete_client_cookie


SECRET_KEY = "a3f9d2e7c1b6f8a0d5e9c4b2a7f1e3d6"  # replace in production
COOKIE_NAME = "streamlit_auth_token"
TOKEN_MAX_AGE = 60 * 60 * 24 * 7


pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def gen_security_stamp() -> str:
    return uuid.uuid4().hex

verify_password = pwd_context.verify
hash_pwd = pwd_context.hash


class AuthHub:
    def __init__(self, db_sess):
        self.signer = TimestampSigner(SECRET_KEY)
        self.cookie_mgr = CookieUtil()
        self.db_sess = db_sess

    def create_token(self, username: str) -> str:
        return self.signer.sign(username.encode()).decode()

    def verify_token(self, token: str) -> str | None:
        try:
            data = self.signer.unsign(token.encode(), max_age=TOKEN_MAX_AGE)
            return data.decode()
        except (BadSignature, SignatureExpired):
            return None
        
    def extract_token(self, raw):
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

    def lookup(self, username: str, password: str):
        usr = self.db_sess.get_user(username)
        if usr and verify_password(password, usr.hashed_password):
            return True
        return False

    def login(self, username: str, password: str) -> bool:
        if self.lookup(username, password):
            new_token = self.create_token(username)
            self.cookie_mgr.set(COOKIE_NAME, new_token)
            st.session_state["auth_user"] = username
            return True
        return False
    
    wait_for_cookie = time.sleep
    
    def logout(self):
        st.session_state.pop("auth_user", None)
        st.session_state["logged_out"] = True
        self.cookie_mgr.delete(COOKIE_NAME)
        delete_client_cookie(COOKIE_NAME)

    def try_authorize_by_cookie(self):
        auth_user = st.session_state.get("auth_user")

        if not auth_user:
            if st.session_state.get("logged_out"):
                # pop the flag and avoid trusting cookies in this run
                st.session_state.pop("logged_out", None)
                token = None
            else:
                token = self.extract_token(self.cookie_mgr.get(COOKIE_NAME))
                if token:
                    verified = self.verify_token(token)
                    if verified:
                        st.session_state["auth_user"] = verified
                        auth_user = verified

        return auth_user

