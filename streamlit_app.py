import streamlit as st
from streamlit.navigation.page import StreamlitPage
from web.utils.authorizer import UserManager
from web.db import connect_db
from web.helper.translator import t
from web.pages.cards import get_all_cards

# ✅ Configure page settings
st.set_page_config(
    page_title="Kepler Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': t("This application helps scientists detect new exoplanet candidates using data from various missions."),
        'Get help': "mailto:admin@fptoj.com",
        'Report a bug': "https://github.com/KienPC1234/ExoVision-Kepler-FPT/issues"
    }
)

st.sidebar.image("static/logo.png", use_container_width=True)


# ✅ Define application pages
ALL_PAGES = [
    st.Page("web/pages/home.py", title=t("Home"), icon="🏠"),
    st.Page("web/pages/Exoplanet_Predictor.py", title=t("Exoplanet Predictor"), icon="🌌"),
    st.Page("web/pages/Exoplanet_Flux_Prediction.py", title=t("Exoplanet Flux Predictor"), icon="💫"),
    st.Page("web/pages/history.py", title=t("History"), icon="📊"),
    st.Page("web/pages/docs.py", title=t("Models Docs"), icon="📄"),
    st.Page("web/pages/helps.py", title=t("Help"), icon="❓"),
    st.Page("web/pages/about.py", title=t("About"), icon="👤"),
]

GUEST_PAGES = [
    st.Page("web/pages/home.py", title=t("Home"), icon="🏠"),
    st.Page("web/pages/helps.py", title=t("Help"), icon="❓"),
    st.Page("web/pages/login.py", title=t("Login With Google"), icon="🔑"),
]


# ✅ Sidebar header (translated)
def render_sidebar_header(user_manager: UserManager) -> None:
    with st.sidebar:
        st.markdown(f"👋 {t('Welcome')}, **{st.user.name}**")
        if st.button(t("Logout"), type="secondary"):
            user_manager.logout()
        st.divider()    


# ✅ Sidebar content preview cards (translated)
def render_sidebar_content(page: StreamlitPage) -> None:
    """Render sidebar content and include a language selector."""
    with st.sidebar.container(height=350):
        # Render cards
        
        cards = get_all_cards()
        current_title = page.title
        if current_title in cards:
            cards[current_title]()
        else:
            cards["Home"]()


# ✅ Dashboard rendering
def dashboard(user_manager: UserManager) -> None:
    page = st.navigation(ALL_PAGES,position="top")
    page.run()
    st.sidebar.divider()
    render_sidebar_header(user_manager)
    render_sidebar_content(page)
    

# ✅ Main entry point
def main() -> None:
    user_manager = UserManager(connect_db())

    with st.sidebar.container():
        # Language selector
        st.markdown("### 🌐 " + t("Language"))

        lang_display = {
            "en": "🇬🇧 English",
            "vi": "🇻🇳 Tiếng Việt",
            "ja": "🇯🇵 日本語",
            "fr": "🇫🇷 Français",
            "es": "🇪🇸 Español",
        }

        # Reverse lookup (so display shows the proper name)
        current_lang = st.session_state.get("preferences", {}).get("lang", "en")
        selected_display = lang_display.get(current_lang, "🇬🇧 English")

        selected_lang_display = st.selectbox(
            label="",
            options=list(lang_display.values()),
            index=list(lang_display.values()).index(selected_display),
            label_visibility="collapsed"
        )

        # Update language in session_state if changed
        new_lang = [k for k, v in lang_display.items() if v == selected_lang_display][0]
        if new_lang != current_lang:
            if "preferences" not in st.session_state:
                st.session_state["preferences"] = {}
            st.session_state["preferences"]["lang"] = new_lang
            st.rerun()

    # Check login state
    if not st.user.is_logged_in:
        page = st.navigation(GUEST_PAGES,position="top")
        if page.title == t("Login With Google"):
            from web.pages import login
            login.main(user_manager)
        else:
            page.run()
        st.stop()

    # ✅ Get or create user and load preferences
    usr = user_manager.get_or_create_user(st.user.email, st.user.name)

    if "preferences" not in st.session_state and hasattr(usr, "preferences"):
        st.session_state["preferences"] = usr.preferencesg

    # Continue to dashboard
    dashboard(user_manager)


if __name__ == "__main__":
    main()
