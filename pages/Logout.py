import streamlit as st

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Logout", layout="centered")

# -------------------------
# Logout Logic
# -------------------------
st.markdown("## 🚪 Logging you out...")

# Clear session state
for key in list(st.session_state.keys()):
    del st.session_state[key]

# Redirect to login page (app.py)
st.switch_page("app.py")
