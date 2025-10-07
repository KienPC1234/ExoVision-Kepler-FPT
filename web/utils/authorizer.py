# Note: I've removed the custom AuthHub class and related cookie management, password hashing, and validation functions
# since we're switching to Streamlit's built-in OIDC authentication. Authentication is now handled externally by the OIDC provider (e.g., Google).
# Signup is managed by the OIDC provider (if configured to allow self-registration). If your provider doesn't support self-signup,
# users must be added manually by the admin in the provider's dashboard.

# To add users to your local DB: After successful login via OIDC, we check if the user exists in the DB by email.
# If not, we create a new User entry. Since no password is needed for OIDC, we set a random hashed password (which won't be used).

# You need to set up secrets.toml as per the documentation, e.g.:
# [auth]
# redirect_uri = "http://localhost:8501/oauth2callback"
# cookie_secret = "xxx"  # Generate a strong secret
# client_id = "xxx"
# client_secret = "xxx"
# server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

# If using multiple providers, adjust as shown in the doc.

# Also, remove or comment out unused imports and functions related to old auth.

import re
import time
import uuid
from passlib.context import CryptContext

import streamlit as st

from web.db.models import User
from web.db import connect_db  # Assuming this is your DB connection function

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
hash_pwd = pwd_context.hash

def gen_security_stamp() -> str:
    return uuid.uuid4().hex

class UserManager:
    def __init__(self, db_sess):
        self.db_sess = db_sess

    def get_or_create_user(self, email: str, name: str = None) -> User:
        usr = self.db_sess.get_user(email)
        if not usr:
            # Create new user with random password (unused for OIDC)
            random_pwd = uuid.uuid4().hex
            usr = User(
                email=email,
                hashed_password=hash_pwd(random_pwd),
                security_stamp=gen_security_stamp()
            )
            if name:
                # Assuming User model has a name field; adjust if not
                usr.name = name
            self.db_sess.create_user(usr)
        return usr

    def logout(self):
        st.logout()
        time.sleep(0.1)
        st.rerun()