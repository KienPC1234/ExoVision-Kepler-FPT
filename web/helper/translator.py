import re
import streamlit as st
from deep_translator import GoogleTranslator

def translate_text(text: str, langid: str = "en") -> str:
    """
    Dịch văn bản sang ngôn ngữ langid (vi, ja, fr, ...).
    Dùng Google Translate miễn phí, không cần API key.
    Hoạt động tốt trong Streamlit.
    """
    if not text:
        return ""
    try:
        return GoogleTranslator(source='en', target=langid).translate(text)
    except Exception as e:
        return f"Translation error: {e}"

def fix_translate(text: str) -> str:
    """Fix formatting error caused by translating markdown"""
    rebuild = ""
    start = 0
    is_start = True
    while (idx := text.find("**", start)) != -1:
        if is_start:
            rebuild += text[start:idx]
        else:
            stripped = text[start:idx].strip()
            if stripped:
                rebuild += "**"
                rebuild += stripped
                rebuild += "**"
        start = idx + 2
        is_start = not is_start
    rebuild += text[start:] 
    return re.subn(r"(?<=\]) *(?=\()", "", rebuild)[0]

@st.cache_resource()
def translate_and_refine(text: str, lang: str):
    return fix_translate(translate_text(text, lang))


def t(text: str) -> str:
    """Translate text if user preference is Vietnamese."""
    lang = st.session_state.get("preferences", {}).get("lang", "en")
    try:
        if lang != "en":
            return translate_and_refine(text, lang)
    except Exception:
        pass
    return text