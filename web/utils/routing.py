import streamlit as st


def redirect(page_name: str):
    """
    Change the current URL to ?page=page_name and force a rerun.
    The target page must read the query param and render itself.
    """
    st.query_params["page"] = page_name
    st.rerun()
