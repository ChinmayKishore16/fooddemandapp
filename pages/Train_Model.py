import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import datetime as dt
import requests
import holidays
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt

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
# Config / Paths
# -------------------------
st.set_page_config(page_title="Train Demand Forecasting Model", layout="wide")

ROOT_DIR = os.getcwd()
MODEL_PATH = os.path.join(ROOT_DIR, "lstm_multioutput_model.h5")
INPUT_SCALER_PATH = os.path.join(ROOT_DIR, "input_scaler.joblib")
OUTPUT_SCALER_PATH = os.path.join(ROOT_DIR, "output_scaler.joblib")
META_PATH = os.path.join(ROOT_DIR, "model_meta.joblib")

WINDOW_SIZE = 7

REQUIRED_COLUMNS = [
    "Date", "item1", "item2", "item3", "item4", "item5",
    "Total_Orders", "Total_Revenue",
    "Holiday", "Weather"
]

# -------------------------
# Utility functions
# -------------------------
def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    return missing, extra

def create_sequences(df, feature_cols, target_cols, window_size=7):
    X_list, y_list, idx_meta = [], [], []
    n = len(df)
    for i in range(n - window_size):
        window = df.iloc[i:i+window_size][feature_cols].values
        target = df.iloc[i+window_size][target_cols].values.astype(float)
        X_list.append(window)
        y_list.append(target)
        idx_meta.append(df.iloc[i+window_size]['Date'])  # keep target date
    return np.array(X_list), np.array(y_list), idx_meta


def scale_X(X, scaler):
    s = X.reshape(-1, X.shape[2])
    s2 = scaler.transform(s)
    return s2.reshape(X.shape)

# -------------------------
# Page UI
# -------------------------
st.title("📂 Train Food Demand Forecasting Model")
st.markdown("Upload your dataset to retrain the **LSTM demand forecasting model**.")

st.warning("**Required Columns:** " + ", ".join(REQUIRED_COLUMNS))
st.markdown("""
- `Date`: parseable by `pd.to_datetime`
- `Holiday`: Yes / No
- `Weather`: Sunny / Rainy / Cold
- `item1..item5`: integer units sold per item
- `Total_Orders` and `Total_Revenue`: numeric
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

epochs = st.slider("Epochs", 5, 100, 30)
batch_size = st.slider("Batch Size", 8, 64, 16)

# Train button
train_button = st.button(" Train Model Now")

# -------------------------
# Main logic
# -------------------------
if uploaded_file is not None and train_button:
    df = pd.read_csv(uploaded_file)
    missing, extra = validate_columns(df)

    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.stop()
    else:
        st.success("✅ Columns validated successfully!")

    # -------------------------
    # Preprocessing
    # -------------------------
    st.subheader("Step 1: Preprocessing")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['DayOfWeek_Num'] = df['Date'].dt.weekday
    df['Holiday'] = df['Holiday'].map({'Yes': 1, 'No': 0})

    df = pd.get_dummies(df, columns=['Weather'])
    for w in ['Weather_Sunny', 'Weather_Rainy', 'Weather_Cold']:
        if w not in df.columns:
            df[w] = 0

    item_units = ['item1','item2','item3','item4','item5']
    target_cols = ['Total_Orders'] + item_units
    feature_cols = item_units + ['DayOfWeek_Num','Holiday'] + [c for c in df.columns if c.startswith("Weather_")]

    # Create sequences
    X, y, idx_meta = create_sequences(df, feature_cols, target_cols, WINDOW_SIZE)

    st.write(f"✅ Generated {X.shape[0]} samples, input shape {X.shape[1:]}")

    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Scaling
    n_features = X_train.shape[2]
    input_scaler = StandardScaler()
    input_scaler.fit(X_train.reshape(-1, n_features))
    X_train_s = scale_X(X_train, input_scaler)
    X_test_s = scale_X(X_test, input_scaler)

    output_scaler = StandardScaler()
    output_scaler.fit(y_train)
    y_train_s = output_scaler.transform(y_train)
    y_test_s = output_scaler.transform(y_test)

    # -------------------------
    # Model Training
    # -------------------------
    st.subheader("Step 2: Training Model")
    model = Sequential()
    model.add(LSTM(128, input_shape=(WINDOW_SIZE, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_train_s.shape[1], activation='linear'))
    model.compile(
    optimizer='adam',
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
    )

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint("best_model.h5", save_best_only=True)

    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_test_s, y_test_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc],
        verbose=0
    )

    st.success("✅ Training Complete!")
    st.line_chart(pd.DataFrame(history.history)[['loss','val_loss']])


    # -------------------------
    # Extra: Tomorrow Prediction Utilities
    # -------------------------

    # Set your API key and location
    OPENWEATHER_API_KEY = "a9123dd1789e5f1aee7a9b8cd6454003"   
    LAT, LON = 12.9716, 77.5946  # Bengaluru coords

    # Indian holidays (example Karnataka)
    STATE_HOLIDAYS = holidays.India(years=[2025], prov="KA")

    def fetch_tomorrow_weather_category(api_key, lat=LAT, lon=LON):
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()

            tomorrow_date = (dt.datetime.utcnow() + dt.timedelta(days=1)).date()
            target_time_utc = dt.datetime.combine(tomorrow_date, dt.time(6, 30))

            closest = min(
                data['list'],
                key=lambda x: abs(dt.datetime.fromtimestamp(x['dt']) - target_time_utc)
            )

            desc = closest['weather'][0]['main'].lower()
            temp = closest['main']['temp']
            precip = closest.get('rain', {}).get('3h', 0)

            if "rain" in desc or precip > 0:
                weather_cat = "Rainy"
            elif temp < 22:
                weather_cat = "Cold"
            else:
                weather_cat = "Sunny"

            forecast_time_used = dt.datetime.fromtimestamp(closest['dt'])
        except Exception as e:
            st.warning(f"⚠️ Weather API failed: {e}, using fallback Sunny.")
            weather_cat, temp, precip = "Sunny", 25, 0
            forecast_time_used = None

        # one-hot vec
        wvec = {c: 0 for c in feature_cols if c.startswith("Weather_")}
        key = f"Weather_{weather_cat}"
        if key in wvec: wvec[key] = 1

        return weather_cat, wvec, temp, precip, forecast_time_used

    def is_holiday_date(date_obj):
        return 1 if date_obj in STATE_HOLIDAYS or date_obj.weekday() >= 5 else 0

    def predict_future(model, input_scaler, output_scaler, df, api_key, days_ahead=1):
    
        # 1. determine target date
        target_date = dt.date.today() + dt.timedelta(days=days_ahead)

        # 2. fetch weather forecast for target date
        weather_cat, wvec, temp, precip, forecast_time_used = fetch_tomorrow_weather_category(api_key)

        # 3. holiday flag
        holiday_flag = is_holiday_date(target_date)

        # 4. prepare last WINDOW_SIZE days of history
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        last_window = df_sorted.iloc[-WINDOW_SIZE:].copy().reset_index(drop=True)

        # replace exogenous features for target day
        last_window.at[WINDOW_SIZE-1, 'DayOfWeek_Num'] = target_date.weekday()
        last_window.at[WINDOW_SIZE-1, 'Holiday'] = holiday_flag
        for c in wvec:
            last_window.at[WINDOW_SIZE-1, c] = wvec[c]

        # 5. prepare input for model
        X_input = last_window[feature_cols].values.reshape(1, WINDOW_SIZE, n_features)
        X_input_s = scale_X(X_input, input_scaler)

        # 6. prediction
        y_pred_s = model.predict(X_input_s)
        y_pred = output_scaler.inverse_transform(y_pred_s)[0]

        # 7. result dict
        result = {target_cols[i]: int(np.round(y_pred[i])) for i in range(len(target_cols))}
        result.update({
            'pred_date': target_date.strftime('%Y-%m-%d'),
            'weather_cat': weather_cat,
            'weather_temp': temp,
            'weather_precip': precip,
            'holiday_flag': int(holiday_flag),
            'forecast_time_used': forecast_time_used.strftime('%Y-%m-%d %H:%M') if forecast_time_used else "Fallback"
        })
        return result

  
    # -------------------------
    # Evaluation
    # -------------------------
    st.subheader("Step 3: Evaluation")
    y_pred = model.predict(X_test_s)
    y_pred_inv = output_scaler.inverse_transform(y_pred)
    y_test_inv = output_scaler.inverse_transform(y_test_s)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)   # manual RMSE calculation
    r2 = r2_score(y_test_inv, y_pred_inv)


    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

        # -------------------------
    # Save Predictions for Reports
    # -------------------------
    test_dates = idx_meta[-len(y_test):]  # align dates for test set

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "Actual": y_test_inv[:, 0],       # Total Orders actual
        "Predicted": y_pred_inv[:, 0]     # Total Orders predicted
    })

    pred_df.to_csv("predictions.csv", index=False)
    st.success("📂 Predictions file saved as predictions.csv for Reports page.")


    # -------------------------
    # Save Model & Metadata
    # -------------------------
    st.subheader("Step 4: Save Model & Rename Items")
    model.save(MODEL_PATH)
    joblib.dump(input_scaler, INPUT_SCALER_PATH)
    joblib.dump(output_scaler, OUTPUT_SCALER_PATH)

    meta = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'window_size': WINDOW_SIZE,
        'n_features': n_features,
        'item_mapping': {f'item{i}': f'item{i}' for i in range(1,6)}
    }
    joblib.dump(meta, META_PATH)

    st.success("✅ Model & scalers saved!")

   # -------------------------
    # Rename items
    # -------------------------
    st.markdown("---")
    st.write("### Rename Food Items")

    with st.form("rename_form", clear_on_submit=False):
        rename_dict = {}
        for i in range(1, 6):
            rename_dict[f'item{i}'] = st.text_input(f"Enter name for item{i}", f"item{i}")

        submitted = st.form_submit_button("Save Names")
        if submitted:
            import joblib
            meta = joblib.load(META_PATH)
            meta['item_mapping'] = rename_dict
            meta['target_cols'] = ['Total_Orders'] + list(rename_dict.values())
            joblib.dump(meta, META_PATH)

            # 🔹 Store in session_state immediately
            st.session_state['item_mapping'] = rename_dict
            st.session_state['target_cols'] = ['Total_Orders'] + list(rename_dict.values())

            st.success("✅ Item names saved! Now reflected across the app.")


    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_KEY_HERE":
        st.info("⚠️ Set your OpenWeather API key in code to enable live tomorrow prediction.")
    else:
        tomorrow_res = predict_future(model, input_scaler, output_scaler, df, OPENWEATHER_API_KEY)
        st.write("📌 Tomorrow's Prediction:")
        st.json(tomorrow_res)

    


