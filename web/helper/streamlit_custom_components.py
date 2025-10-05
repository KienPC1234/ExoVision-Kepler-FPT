import streamlit.components.v1 as components
import json
import pandas as pd

def embed_iframe(url: str, json_data: dict, height: int = 600, width: str = "100%"):
    json_str = json.dumps(json_data)
    iframe_code = f"""
    <iframe id="planetIframe" src="{url}" width="{width}" height="600" style="border:none;"></iframe>
    <script>
        const iframe = document.getElementById('planetIframe');
        iframe.onload = () => {{
            iframe.contentWindow.postMessage({json_str}, "*"); 
        }};
    </script>
    """
    components.html(iframe_code, height=height)