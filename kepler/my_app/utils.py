import datetime
from typing import Union, Optional, Literal

import streamlit as st
import streamlit.components.v1 as components
import extra_streamlit_components as stx


def redirect(page_name: str):
    """
    Change the current URL to ?page=page_name and force a rerun.
    The target page must read the query param and render itself.
    """
    st.query_params["page"] = page_name
    st.rerun()

def delete_client_cooke(cookie_name):
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



class CookieUtil:
    manager: stx.CookieManager

    def __init__(self):
        self.manager = stx.CookieManager()
        self._set = self.manager.set
        self._delete = self.manager.delete
        self.get = self.manager.get

    def set(
            self,
            cookie: str,
            val: Union[str, int, float, bool],
            path: str = "/",
            max_age: Optional[float] = None,
            secure: Optional[bool] = False,
            same_site: Union[bool, None, Literal["lax", "strict"]] = "lax",
            **kwargs
        ):
        """
        Set cookie with options. Use the same options for set/delete to avoid mismatches.
        For local dev use secure=False. In production (HTTPS) set secure=True.
        """
        self._set(
            cookie,
            val=val,
            max_age=max_age,
            path=path,
            secure=secure,
            same_site=same_site,
            **kwargs
        )

    def delete(self, cookie: str):
        try:
            self._delete(cookie)
        except KeyError:
            pass