import re
import time
import uuid
from passlib.context import CryptContext

import streamlit as st
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

from .cookie import CookieUtil, delete_client_cookie
from ..db.models import User


SECRET_KEY = "a3f9d2e7c1b6f8a0d5e9c4b2a7f1e3d6"  # replace in production
AUTH_TOKEN_COOKIE = "streamlit_auth_token"
REVISION_COOKIE = "streamlit_security_stamp"
TOKEN_MAX_AGE = 60 * 60 * 24 * 7


pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

def gen_security_stamp() -> str:
    return uuid.uuid4().hex

verify_password = pwd_context.verify
hash_pwd = pwd_context.hash


def validate_password(password: str) -> tuple[bool, str]:
    """
    Validates password strength using following rules:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one number
    - At least one special character
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    
    return True, "Password meets all requirements"

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

    def authenticated_session(self, usr: User):
        """Create an authenticated session for the user."""
        new_token = self.create_token(usr.email)
        self.cookie_mgr.set(
            AUTH_TOKEN_COOKIE, 
            new_token, 
            key=f"set_auth_{usr.email}"
        )
        self.cookie_mgr.set(
            REVISION_COOKIE, 
            usr.security_stamp, 
            key=f"set_stamp_{usr.email}"
        )
        st.session_state["auth_user"] = usr.email

    def lookup(self, username: str, password: str):
        usr = self.db_sess.get_user(username)
        if usr and verify_password(password, usr.hashed_password):
            return usr
        return None

    def login(self, username: str, password: str) -> bool:
        if usr := self.lookup(username, password):
            self.authenticated_session(usr)
            return True
        return False
    
    def signup(self, username: str, password: str) -> bool:
        if self.db_sess.is_username_taken(username):
            return False
        
        new_usr = User(
            email=username,
            hashed_password=hash_pwd(password)
        )
        self.db_sess.create_user(new_usr)

        # Automatically login
        self.authenticated_session(new_usr)
        return True
    
    wait_for_cookie = time.sleep

    def logout(self):
        """Clear the authenticated session."""
        username = st.session_state.pop("auth_user", None)
        st.session_state["logged_out"] = True
        self.cookie_mgr.delete(
            AUTH_TOKEN_COOKIE, 
            key=f"del_auth_{username if username else 'anon'}"
        )
        self.cookie_mgr.delete(
            REVISION_COOKIE, 
            key=f"del_stamp_{username if username else 'anon'}"
        )
        delete_client_cookie(AUTH_TOKEN_COOKIE)
        delete_client_cookie(REVISION_COOKIE)

    def try_authorize_by_cookie(self):
        auth_user = st.session_state.get("auth_user")

        if not auth_user:
            if st.session_state.get("logged_out"):
                # pop the flag and avoid trusting cookies in this run
                st.session_state.pop("logged_out", None)
                token = None
            else:
                token = self.cookie_mgr.get(AUTH_TOKEN_COOKIE)
                if token:
                    verified = self.verify_token(token)
                    if verified \
                        and (usr := self.db_sess.get_user(verified)) \
                        and usr.security_stamp == self.cookie_mgr.get(REVISION_COOKIE):

                        st.session_state["auth_user"] = verified
                        auth_user = verified

        return auth_user

