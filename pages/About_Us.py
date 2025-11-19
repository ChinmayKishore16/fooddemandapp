import streamlit as st

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
st.set_page_config(page_title="About Us - Food Demand Forecast", layout="wide")

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
    <style>
    .about-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .team-card {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem;
        text-align: center;
        color: #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stat-card {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
st.markdown("<h1 style='text-align: center; color: #667eea;'>🛈 About Our Food Demand Forecasting System</h1>", unsafe_allow_html=True)

# -------------------------
# Mission Section
# -------------------------
st.markdown("""
    <div class="about-container">
        <h2>🎯 Our Mission</h2>
        <p style="font-size: 18px; line-height: 1.6;">
        We are revolutionizing the restaurant industry through AI-powered demand forecasting. 
        Our mission is to help restaurants reduce food waste, optimize inventory management, 
        and increase profitability while ensuring customer satisfaction.
        </p>
    </div>
""", unsafe_allow_html=True)

# -------------------------
# Key Features
# -------------------------
st.markdown("## 🚀 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI-Powered Predictions</h3>
            <p>Advanced LSTM neural networks analyze historical data, weather patterns, 
            and holiday schedules to provide accurate demand forecasts.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <h3>📊 Real-time Analytics</h3>
            <p>Comprehensive reporting dashboard with sales trends, performance metrics, 
            and actionable insights for data-driven decisions.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <h3>🌤️ Weather Integration</h3>
            <p>Live weather data integration to predict how weather conditions 
            impact customer demand and food preferences.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="feature-card">
            <h3>🎯 Customizable Models</h3>
            <p>Train and retrain models with your own data to ensure predictions 
            are tailored to your specific restaurant's patterns.</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Statistics
# -------------------------
st.markdown("## 📈 Impact Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="stat-card">
            <h3>35%</h3>
            <p>Food Waste Reduction</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="stat-card">
            <h3>25%</h3>
            <p>Cost Savings</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="stat-card">
            <h3>90%</h3>
            <p>Prediction Accuracy</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="stat-card">
            <h3>500+</h3>
            <p>Restaurants Served</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Technology Stack
# -------------------------
st.markdown("## 🛠️ Technology Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
        *Frontend & UI:*
        - 🎨 Streamlit
        - 📱 Responsive Design
        - 🎯 User-friendly Interface
    """)

with tech_col2:
    st.markdown("""
        *Machine Learning:*
        - 🧠 TensorFlow/Keras
        - 📊 Scikit-learn
        - 🔄 LSTM Neural Networks
    """)

with tech_col3:
    st.markdown("""
        *Data & APIs:*
        - 🌤️ OpenWeather API
        - 📈 Pandas & NumPy
        - 🗄️ SQLite Database
    """)

# -------------------------
# Team Section
# -------------------------
st.markdown("## 👥 Our Team")

team_col1, team_col2, team_col3 = st.columns(3)

with team_col1:
    st.markdown("""
        <div class="team-card">
            <h3>👨‍💻 AI/ML Engineers</h3>
            <p>Specialists in deep learning, neural networks, and predictive analytics 
            with expertise in restaurant industry challenges.</p>
        </div>
    """, unsafe_allow_html=True)

with team_col2:
    st.markdown("""
        <div class="team-card">
            <h3>👨‍🍳 Industry Experts</h3>
            <p>Restaurant professionals with deep understanding of food service 
            operations, inventory management, and customer behavior.</p>
        </div>
    """, unsafe_allow_html=True)

with team_col3:
    st.markdown("""
        <div class="team-card">
            <h3>🎨 UX/UI Designers</h3>
            <p>Creating intuitive and beautiful interfaces that make complex 
            data accessible to restaurant managers and staff.</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------
# Contact Information
# -------------------------
st.markdown("## 📞 Contact Us")

contact_col1, contact_col2 = st.columns(2)

with contact_col1:
    st.markdown("""
        *🏢 Office Address:*
        Food Demand Solutions Ltd.
        Tech Park, Bangalore
        Karnataka, India - 560001
        
        *📧 Email:*
        info@fooddemandai.com
        support@fooddemandai.com
    """)

with contact_col2:
    st.markdown("""
        *📱 Phone:*
        +91 80 1234 5678
        +91 80 8765 4321
        
        *🌐 Website:*
        www.fooddemandai.com
        
        *📱 Social Media:*
        @FoodDemandAI
    """)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea, #764ba2); 
                border-radius: 10px; color: white; margin-top: 2rem;">
        <h3>🍽️ Transforming Restaurants with AI</h3>
        <p style="font-size: 16px;">
        Join hundreds of restaurants already using our AI-powered forecasting system 
        to reduce waste, increase profits, and improve customer satisfaction.
        </p>
        <p style="font-size: 14px; opacity: 0.8;">
        © 2025 Food Demand Solutions. All rights reserved. | Built with ❤️ using Streamlit
        </p>
    </div>
""", unsafe_allow_html=True)