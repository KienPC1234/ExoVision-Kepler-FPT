import datetime, time
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



class CookieUtil:
    manager: stx.CookieManager

    def __init__(self):
        self.manager = stx.CookieManager()
        self._set = self.manager.set
        self._delete = self.manager.delete

        # Populate cookies early — some environments populate asynchronously.
        try:
            self.manager.get_all()
            time.sleep(0.05)
        except Exception:
            pass

    def get(self, cookie: str):
        """
        Safe get: if the component hasn't populated cookies yet it may have
        'cookies' set to a boolean flag. We try to initialize/populate and
        fall back to None if we can't get a value.
        """
        # fast path: if manager has a dict-like cookies attribute
        cookies = getattr(self.manager, "cookies", None)
        if isinstance(cookies, dict):
            return cookies.get(cookie)

        # last attempt: force get_all(), wait briefly, then inspect cookies again
        try:
            self.manager.get_all()
            time.sleep(0.05)
            if isinstance(self.manager.cookies, dict):
                return self.manager.get(cookie)
        except Exception:
            pass

        return None

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
        # delegate to bound manager.set
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