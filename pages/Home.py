import streamlit as st
import sqlite3

# -------------------------
# Sidebar (custom links, exclude app.py)
# -------------------------
st.markdown("""
    <style>
    /* Hide Streamlit's default sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)


st.sidebar.title("📂 Navigation")
st.sidebar.page_link("pages/Home.py", label=" Home")
st.sidebar.page_link("pages/Food_Demand.py", label="🍽 Food Demand")
st.sidebar.page_link("pages/Reports.py", label="📑  Reports")
st.sidebar.page_link("pages/Train_Model.py", label="🛠 Train Model")
st.sidebar.page_link("pages/About_Us.py", label="🛈 About Us")
st.sidebar.page_link("pages/Logout.py", label="🚪 Logout")


# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="Home - Food Demand Forecast System", layout="centered")

# -------------------------
# DB Helper to Fetch Restaurant Details
# -------------------------
def get_restaurant_name():
    try:
        conn = sqlite3.connect("users.db")   # same DB used at registration
        cur = conn.cursor()
        cur.execute("SELECT restaurant FROM users LIMIT 1;")  
        row = cur.fetchone()
        conn.close()
        if row:
            return row[0]
        else:
            return "Demo Restaurant"
    except Exception as e:
        return f"Error fetching name ({e})"

# -------------------------
# Page Layout Styling
# -------------------------
st.markdown("""
    <style>
    .centered {
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 40px;
        color: #FFD700;  /* golden yellow */
    }
   
    .predict-btn {
        display: inline-block;
        padding: 15px 38px;
        font-size: 22px;
        font-weight: bold;
        color: black !important;
        background-color: #E5B001;
        border-radius: 10px;
        text-decoration: none;
        transition: 0.3s;
    }
    .predict-btn:hover {
        background-color: #DC8D04;
        color: black !important;
    }
    .card {
        background: rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-top: 30px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Title & Subtitle
# -------------------------
st.markdown("<h1 class='centered'>✨ Welcome to Food Demand Forecasting System ✨</h1>", unsafe_allow_html=True)
st.markdown("<h4 class='subtitle'>AI-powered predictions to reduce waste and optimize restaurant operations</h4>", unsafe_allow_html=True)

# -------------------------
# Predict Button (center, styled)
# -------------------------
st.markdown(
    """
    <div style='text-align:center; margin-bottom:50px; padding: 15px 40px; cursor: pointer; transition: 0.3s;'>
        <a href='/Food_Demand' target='_self' class='predict-btn'>
             Predict Food Demand
        </a>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------
# Restaurant Details Card
# -------------------------
rest_name = get_restaurant_name()
st.markdown(f"""
    <div class="card">
        <h2>🏨 Restaurant Details</h2>
        <p><b>Name:</b> {rest_name}</p>
        <p><b>📍 Location:</b> Bengaluru, India</p>
        <p><b>🍽 Cuisine:</b> Multi-Cuisine</p>
        <p><b>⭐ Rating:</b> 4.5</p>
        <hr>
        <p><i>“Our restaurant is committed to sustainability and innovation.  
        With AI-driven demand forecasting, we are reducing food waste, improving efficiency,  
        and ensuring customer satisfaction.”</i></p>
    </div>
""", unsafe_allow_html=True)
