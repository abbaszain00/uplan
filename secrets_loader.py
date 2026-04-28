import os
try:
    import streamlit as st
    os.environ.setdefault("OPENROUTER_API_KEY", st.secrets.get("OPENROUTER_API_KEY", ""))
    os.environ.setdefault("TICKETMASTER_API_KEY", st.secrets.get("TICKETMASTER_API_KEY", ""))
except Exception:
    pass  # Running outside Streamlit — use env vars directly