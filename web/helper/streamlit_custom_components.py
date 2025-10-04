import streamlit as st
import streamlit.components.v1 as components

def embed_iframe(url: str, height: int = 600, width: str = "100%"):
    """
    Nhúng iframe vào Streamlit app.
    
    Parameters:
    - url: Đường dẫn đến trang web cần nhúng.
    - height: Chiều cao iframe (px).
    - width: Chiều rộng iframe (mặc định là 100%).
    """
    iframe_code = f"""
        <iframe src="{url}" width="{width}" height="{height}" style="border:none;" allow="fullscreen"></iframe>
    """
    components.html(iframe_code, height=height)
