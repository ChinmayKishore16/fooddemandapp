import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import joblib
import requests
import holidays
from tensorflow.keras.models import load_model

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
# Config & Paths
# -------------------------
st.set_page_config(page_title="Food Demand Prediction", layout="wide")



MODEL_PATH = "lstm_multioutput_model.h5"
INPUT_SCALER_PATH = "input_scaler.joblib"
OUTPUT_SCALER_PATH = "output_scaler.joblib"
META_PATH = "model_meta.joblib"

# OpenWeather API
OPENWEATHER_API_KEY = "a9123dd1789e5f1aee7a9b8cd6454003"
LAT, LON = 12.9716, 77.5946  # Bengaluru
STATE_HOLIDAYS = holidays.India(years=[2025], prov="KA")

# -------------------------
# Custom CSS Styling
# -------------------------
st.markdown(
    """
    <style>
    /* Center all content */
    .block-container {
        max-width: 900px;
        margin: auto;
        text-align: center;
    }
    /* Enlarge radio buttons */
    div[data-baseweb="radio"] label {
        font-size: 40px !important;
        font-weight: bold;
        margin-right: 25px;
        text-align: center;
    }
    /* Big yellow prediction numbers */
    .prediction-number {
        font-size: 28px;
        font-weight: bold;
        color: #FFD700; /* gold/yellow */
    }

    /* Center the radio buttons */
    div[role="radiogroup"] {
        justify-content: center;
    }



    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load model, scalers, metadata
# -------------------------
@st.cache_resource
def load_all():
    model = load_model(MODEL_PATH, compile=False)
    input_scaler = joblib.load(INPUT_SCALER_PATH)
    output_scaler = joblib.load(OUTPUT_SCALER_PATH)
    meta = joblib.load(META_PATH)
    return model, input_scaler, output_scaler, meta

model, input_scaler, output_scaler, meta = load_all()
feature_cols = meta['feature_cols']
target_cols = meta['target_cols']
window_size = meta['window_size']
item_mapping = meta['item_mapping']

# -------------------------
# Weather & Holiday Helpers
# -------------------------
def fetch_weather_for_date(api_key, lat=LAT, lon=LON, days_ahead=1):
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        target_date = (dt.datetime.utcnow() + dt.timedelta(days=days_ahead)).date()
        target_time_utc = dt.datetime.combine(target_date, dt.time(6, 30))

        closest = min(
            data['list'],
            key=lambda x: abs(dt.datetime.fromtimestamp(x['dt']) - target_time_utc)
        )
        desc = closest['weather'][0]['main'].lower()
        temp = closest['main']['temp']
        precip = closest.get('rain', {}).get('3h', 0)

        if "rain" in desc or precip > 0:
            return "Rainy", temp
        elif temp < 22:
            return "Cold", temp
        else:
            return "Sunny", temp
    except Exception as e:
        st.warning(f"⚠️ Weather API failed, fallback Sunny. {e}")
        return "Sunny", 25

def is_holiday(date_obj):
    return "Yes" if date_obj in STATE_HOLIDAYS or date_obj.weekday() >= 5 else "No"

# -------------------------
# Helper: Prepare model input
# -------------------------
def prepare_input(df_history, target_date, input_scaler, weather_cat, holiday_flag):
    df = df_history.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    if 'DayOfWeek_Num' not in df.columns:
        df['DayOfWeek_Num'] = df['Date'].dt.weekday
    if df['Holiday'].dtype == object:
        df['Holiday'] = df['Holiday'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, columns=['Weather'])
    for w in ['Weather_Sunny','Weather_Rainy','Weather_Cold']:
        if w not in df.columns:
            df[w] = 0

    # Inject target date’s exogenous features in last row
    df = df.sort_values("Date").reset_index(drop=True)
    df.loc[df.index[-1], 'DayOfWeek_Num'] = target_date.weekday()
    df.loc[df.index[-1], 'Holiday'] = 1 if holiday_flag == "Yes" else 0
    for w in ['Weather_Sunny','Weather_Rainy','Weather_Cold']:
        df.loc[df.index[-1], w] = 1 if w == f"Weather_{weather_cat}" else 0

    # Scale input
    X = df[feature_cols].values
    X_scaled = input_scaler.transform(X)
    last_seq = X_scaled[-window_size:]
    return np.expand_dims(last_seq, axis=0)

# -------------------------
# UI
# -------------------------
st.title("🍽️ Food Demand Prediction")

menu_option = st.radio(
    "Choose Prediction Type",
    ["Whole Menu Prediction", "Individual Item Prediction"],
    horizontal=True
)

# Example history data
try:
    df_history = pd.read_csv("restaurant_sales_updated_items.csv")
    df_history['Date'] = pd.to_datetime(df_history['Date'])
except:
    st.error("⚠️ Could not load history CSV. Please ensure restaurant_sales_updated_items.csv exists.")
    st.stop()

today = dt.date.today()
min_date = today
max_date = today + dt.timedelta(days=7)  # Allow only 7-day forecast

# -------------------------
# Whole Menu Prediction
# -------------------------
if menu_option == "Whole Menu Prediction":
    st.subheader("📊 Whole Menu Demand Forecast")
    target_date = st.date_input("Select Date", min_value=min_date, max_value=max_date)

    if st.button("🔮 Predict Whole Menu"):
        days_ahead = (target_date - today).days
        weather_cat, temp = fetch_weather_for_date(OPENWEATHER_API_KEY, days_ahead=days_ahead)
        holiday_flag = is_holiday(target_date)

        X_input = prepare_input(df_history, target_date, input_scaler, weather_cat, holiday_flag)
        y_pred = model.predict(X_input)
        y_pred_inv = output_scaler.inverse_transform(y_pred)[0]

        st.markdown(f"### 📅 {target_date}")
        st.markdown(f"🌤️ **Weather:** {weather_cat} ({temp}°C)")
        st.markdown(f"🏖️ **Holiday:** {holiday_flag}")

        st.markdown("## 🛒 Predicted Orders")
        for i, col in enumerate(target_cols):
            st.markdown(
                f"<div class='prediction-number'>{col}: {int(round(y_pred_inv[i]))} units</div>",
                unsafe_allow_html=True
            )

# -------------------------
# Individual Item Prediction
# -------------------------
elif menu_option == "Individual Item Prediction":
    st.subheader("🥗 Individual Item Demand Forecast")
    target_date = st.date_input("Select Date", min_value=min_date, max_value=max_date)
    item_choice = st.selectbox("Select Food Item", list(item_mapping.values()))

    if st.button("🔮 Predict Item"):
        days_ahead = (target_date - today).days
        weather_cat, temp = fetch_weather_for_date(OPENWEATHER_API_KEY, days_ahead=days_ahead)
        holiday_flag = is_holiday(target_date)

        X_input = prepare_input(df_history, target_date, input_scaler, weather_cat, holiday_flag)
        y_pred = model.predict(X_input)
        y_pred_inv = output_scaler.inverse_transform(y_pred)[0]

        idx = target_cols.index(item_choice)
        pred_units = int(round(y_pred_inv[idx]))

        st.markdown(f"### 📅 {target_date}")
        st.markdown(f"🌤️ **Weather:** {weather_cat} ({temp}°C)")
        st.markdown(f"🏖️ **Holiday:** {holiday_flag}")

        st.markdown(
            f"<div class='prediction-number'>Predicted demand for {item_choice}: {pred_units} units</div>",
            unsafe_allow_html=True
        )
