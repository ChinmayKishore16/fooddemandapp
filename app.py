import streamlit as st
import sqlite3
import hashlib

# ======================
# Database Setup
# ======================
conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    restaurant TEXT UNIQUE,
    phone TEXT,
    password TEXT
)
""")
conn.commit()

# ======================
# Helper Functions
# ======================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(name, restaurant, phone, password):
    c.execute("INSERT INTO users (name, restaurant, phone, password) VALUES (?, ?, ?, ?)",
              (name, restaurant, phone, hash_password(password)))
    conn.commit()

def login_user(restaurant, password):
    c.execute("SELECT * FROM users WHERE restaurant=? AND password=?",
              (restaurant, hash_password(password)))
    return c.fetchone()

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Food Demand Forecasting System", layout="centered")

# Hide sidebar completely on this page
hide_sidebar_style = """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Glassmorphic style
st.markdown("""
    <style>
    body {
        background: #121212;
        color: #fff;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🍽️ Food Demand Forecasting System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Restaurant Admin Login</h3>", unsafe_allow_html=True)

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "restaurant" not in st.session_state:
    st.session_state.restaurant = ""

# ======================
# Login / Register Tabs
# ======================
tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Login")
    restaurant = st.text_input("Restaurant Name", key="login_restaurant")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_button"):
        user = login_user(restaurant, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.restaurant = restaurant
            st.success("✅ Login successful!")
            st.switch_page("pages/Home.py")   # redirect to home page
        else:
            st.error("❌ Invalid credentials")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Register New Admin")
    name = st.text_input("Name", key="register_name")
    restaurant_new = st.text_input("Restaurant Name (unique)", key="register_restaurant")
    phone = st.text_input("Phone Number", key="register_phone")
    password_new = st.text_input("Password", type="password", key="register_password")

    if st.button("Register", key="register_button"):
        try:
            add_user(name, restaurant_new, phone, password_new)
            st.success("✅ Registered successfully! Please login.")
        except sqlite3.IntegrityError:
            st.error("⚠️ Restaurant name already exists. Choose another.")
    st.markdown("</div>", unsafe_allow_html=True)
