import streamlit as st
import streamlit.components.v1 as components



def redirect(page_name: str):
    """
    Change the current URL to ?page=page_name and force a rerun.
    The target page must read the query param and render itself.
    """
    st.query_params["page"] = page_name
    st.rerun()

def delete_client_cookie(cookie_name):
    """
    Inject small JS snippet that:
      - attempts to delete the cookie for several domain variants
      - waits a short moment then reloads the page (client-side reload)
    This guarantees the new request won't include the cookie.
    Note: JS cannot delete HttpOnly cookies.
    """
    js = f"""
    <script>
    (function() {{
      function deleteCookie(name, path, domain) {{
        var cookie = name + "=; Max-Age=0; path=" + path + ";";
        if (domain) cookie += " domain=" + domain + ";";
        document.cookie = cookie;
        // old expiration style
        document.cookie = name + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=" + path + (domain ? "; domain=" + domain : "") + ";";
      }}

      var name = "{cookie_name}";

      // 1) Try straightforward deletion for current hostname
      deleteCookie(name, "/", location.hostname);

      // 2) Try deleting on parent domain variants (example: example.com, .example.com)
      var parts = location.hostname.split('.');
      for (var i = 0; i < parts.length - 1; i++) {{
        var dom = parts.slice(-1 - i).join('.');
        deleteCookie(name, "/", dom);
        deleteCookie(name, "/", "." + dom);
      }}

      // 3) Give browser a short moment to apply removal, then reload the top window.
      setTimeout(function(){{ window.location.reload(); }}, 180);
    }})();
    </script>
    """
    # minimal height so it renders but doesn't add UI noise
    components.html(js, height=10)