import streamlit as st
from ..utils.authorizer import UserManager  # renamed from authorizer
from ..utils.routing import redirect

def main(user_manager: UserManager):
    # --- Load user preferences (language) ---
    lang = "en"
    if "preferences" in st.session_state:
        lang = st.session_state["preferences"].get("lang", "en")

    # --- Language dictionary ---
    TEXT = {
        "en": {
            "title": "🌌 ExoVision Dashboard Login",
            "welcome": (
                "Welcome to the **ExoVision Dashboard!**  \n"
                "Sign in with your Google account to explore exoplanets, predictions, and more."
            ),
            "login_btn": "🔑 Log in with Google",
        },
        "vi": {
            "title": "🌌 Đăng nhập Bảng điều khiển ExoVision",
            "welcome": (
                "Chào mừng bạn đến với **Bảng điều khiển ExoVision!**  \n"
                "Đăng nhập bằng tài khoản Google để khám phá các hành tinh ngoài hệ mặt trời, "
                "các dự đoán và nhiều điều thú vị khác."
            ),
            "login_btn": "🔑 Đăng nhập bằng Google",
        },
    }

    t = TEXT.get(lang, TEXT["en"])  # fallback to English

    # --- UI ---
    st.title(t["title"])
    st.markdown(t["welcome"])

    if st.button(t["login_btn"], type="primary"):
        st.login()  # Or st.login("google") if multiple

    st.stop()
