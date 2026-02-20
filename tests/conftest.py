"""pytest configuration and shared fixtures.

Stubs out optional heavy dependencies (streamlit) so that UI backend
tests can run without requiring the streamlit package to be installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _make_streamlit_stub() -> types.ModuleType:
    """Build a minimal streamlit stub that satisfies backend.py's import-time usage.

    backend.py uses ``import streamlit as st`` at module level, then decorates
    functions with ``@st.cache_data``.  The stub provides:
      - st.cache_data(show_spinner=...) — returns a no-op decorator that also
        sets __wrapped__ on the decorated function so tests can call the raw impl.
    """
    st = types.ModuleType("streamlit")

    class _CacheData:
        def __call__(self, fn=None, *, show_spinner=True, **_kw):
            if fn is None:
                # Called with keyword args: @st.cache_data(show_spinner=False)
                def decorator(f):
                    f.__wrapped__ = f
                    return f
                return decorator
            # Called bare: @st.cache_data
            fn.__wrapped__ = fn
            return fn

        def clear(self):
            pass

    st.cache_data = _CacheData()

    # Stub out any other st.* attributes that may be referenced at import time
    for attr in ("error", "warning", "info", "success", "stop", "spinner",
                 "session_state", "sidebar", "title", "caption", "divider",
                 "header", "subheader", "dataframe", "json", "metric",
                 "columns", "radio", "selectbox", "text_input", "number_input",
                 "multiselect", "toggle", "button", "expander", "page_link",
                 "bar_chart", "image", "markdown", "write", "code"):
        setattr(st, attr, MagicMock())

    return st


# Inject the stub before any test module imports weather_tool.ui.*
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = _make_streamlit_stub()
