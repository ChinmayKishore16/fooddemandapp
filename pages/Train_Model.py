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

def load_existing_model_info():
    """Load information about previously trained models"""
    try:
        if (os.path.exists(MODEL_PATH) and 
            os.path.exists(META_PATH) and 
            os.path.exists("predictions.csv")):
            
            meta = joblib.load(META_PATH)
            
            # Try to load training metrics if available
            training_info = {
                'model_exists': True,
                'model_path': MODEL_PATH,
                'meta_info': meta,
                'has_predictions': os.path.exists("predictions.csv")
            }
            
            # Load predictions to show basic info
            if training_info['has_predictions']:
                pred_df = pd.read_csv("predictions.csv")
                training_info['prediction_samples'] = len(pred_df)
            
            return training_info
        else:
            return {'model_exists': False}
    except Exception as e:
        return {'model_exists': False, 'error': str(e)}

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
st.markdown("Upload your dataset to retrain the *LSTM demand forecasting model*.")

st.warning("*Required Columns:* " + ", ".join(REQUIRED_COLUMNS))
st.markdown("""
- Date: parseable by pd.to_datetime
- Holiday: Yes / No
- Weather: Sunny / Rainy / Cold
- item1..item5: integer units sold per item
- Total_Orders and Total_Revenue: numeric
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

epochs = st.slider("Epochs", 5, 200, 80)
batch_size = st.slider("Batch Size", 8, 64, 16)

# Show previous training results if available
model_info = load_existing_model_info()

# Check both session state and disk for training results
if ('training_complete' in st.session_state and st.session_state.training_complete) or model_info['model_exists']:
    
    # If we have session state results, use those; otherwise show basic model info
    if 'training_complete' in st.session_state and st.session_state.training_complete:
        st.success("✅ Model training completed successfully!")
        results = st.session_state.model_results
        
        # Display detailed training results
        st.markdown("## 📊 Training Results")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE (Mean Absolute Error)", f"{results['mae']:.2f}")
        with col2:
            st.metric("RMSE (Root Mean Squared Error)", f"{results['rmse']:.2f}")
        with col3:
            st.metric("R² (Coefficient of Determination)", f"{results['r2']:.3f}")
        
        # Additional info
        st.info(f"Generated {results['samples_count']} samples with input shape {results['input_shape']}")
        
        # Show training history if available
        if 'training_history' in st.session_state:
            st.markdown("### 📈 Training History")
            history_df = pd.DataFrame(st.session_state.training_history)
            st.line_chart(history_df[['loss', 'val_loss']])
        
        # Show predictions vs actual if available
        if 'evaluation_results' in st.session_state:
            eval_data = st.session_state.evaluation_results
            st.markdown("### 🎯 Model Performance")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Actual': eval_data['y_test_inv'][:, 0][:20],  # First 20 samples
                'Predicted': eval_data['y_pred_inv'][:, 0][:20]
            })
            
            st.write("*Actual vs Predicted (First 20 Test Samples):*")
            st.line_chart(comparison_df)
        
        # Show tomorrow forecast if available
        if 'tomorrow_forecast' in st.session_state:
            st.markdown("### 🔮 Tomorrow's Forecast Preview")
            forecast_data = st.session_state.tomorrow_forecast
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"*📅 Date:* {forecast_data['pred_date']}")
                st.write(f"*🌤️ Weather:* {forecast_data['weather_cat']} ({forecast_data['weather_temp']}°C)")
                st.write(f"*🏖️ Holiday:* {'Yes' if forecast_data['holiday_flag'] else 'No'}")
            
            with col2:
                st.write("*📊 Predicted Demand:*")
                for item, value in forecast_data.items():
                    if item not in ['pred_date', 'weather_cat', 'weather_temp', 'weather_precip', 'holiday_flag', 'forecast_time_used']:
                        st.write(f"• {item}: *{value} units*")
    
    elif model_info['model_exists']:
        # Show basic model info when session state is lost but model files exist
        st.success("✅ Previously trained model found!")
        st.markdown("## 📊 Model Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model Status", "Ready")
        with col2:
            if 'prediction_samples' in model_info:
                st.metric("Prediction Samples", model_info['prediction_samples'])
        with col3:
            meta_info = model_info['meta_info']
            st.metric("Features", len(meta_info.get('feature_cols', [])))
        
        st.info("ℹ️ A trained model is available. You can use it for predictions or train a new one.")
        
        # Show current item mapping
        if 'meta_info' in model_info and 'item_mapping' in model_info['meta_info']:
            st.markdown("### 🏷️ Current Item Names")
            item_mapping = model_info['meta_info']['item_mapping']
            for key, value in item_mapping.items():
                st.write(f"• {key} → *{value}*")
        
        # Show predictions chart if available
        if model_info['has_predictions']:
            st.markdown("### 📈 Previous Predictions")
            try:
                pred_df = pd.read_csv("predictions.csv")
                pred_df['Date'] = pd.to_datetime(pred_df['Date'])
                
                # Show last 20 predictions
                recent_pred = pred_df.tail(20)
                st.line_chart(recent_pred.set_index('Date')[['Actual', 'Predicted']])
                st.write(f"📊 Showing last 20 predictions from {len(pred_df)} total samples")
            except Exception as e:
                st.warning(f"Could not load prediction chart: {str(e)}")
    
    st.write("✨ You can now use the 'Food Demand' page for predictions!")
    st.markdown("---")

# Train button
if st.button("🚀 Train Model Now"):
    if uploaded_file is not None:
        st.session_state.start_training = True
    else:
        st.error("❌ Please upload a CSV file first before training the model!")
        st.warning("📁 Use the file uploader above to select your training data.")

# Initialize session state
if 'start_training' not in st.session_state:
    st.session_state.start_training = False
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# -------------------------
# Main Training Logic
# -------------------------
if uploaded_file is not None and st.session_state.start_training:
    
    # Clear previous training results
    if 'model_results' in st.session_state:
        del st.session_state.model_results
    if 'training_history' in st.session_state:
        del st.session_state.training_history
    if 'evaluation_results' in st.session_state:
        del st.session_state.evaluation_results
    if 'tomorrow_forecast' in st.session_state:
        del st.session_state.tomorrow_forecast
    
    df = pd.read_csv(uploaded_file)
    missing, extra = validate_columns(df)

    if missing:
        st.error(f"❌ Missing columns: {missing}")
        st.session_state.start_training = False
        st.stop()
    else:
        st.success("✅ Columns validated successfully!")

    # Store data in session state
    st.session_state.training_data = df.copy()

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

    # Clear any existing models from memory
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization

    tf.keras.backend.clear_session()

    # Build improved model
    with tf.name_scope("demand_forecasting_model"):
        model = Sequential(name="lstm_demand_model")
        model.add(LSTM(128, return_sequences=True, input_shape=(WINDOW_SIZE, n_features), name="lstm_layer_1"))
        model.add(Dropout(0.3, name="dropout_1"))
        model.add(LSTM(64, return_sequences=False, name="lstm_layer_2"))
        model.add(BatchNormalization(name="batch_norm"))
        model.add(Dropout(0.3, name="dropout_2"))
        model.add(Dense(64, activation='relu', name="dense_hidden_1"))
        model.add(Dense(32, activation='relu', name="dense_hidden_2"))
        model.add(Dense(y_train_s.shape[1], activation='linear', name="output_layer"))

        # Use Huber Loss (more robust than MSE for sales data)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber(),
            metrics=[MeanAbsoluteError()]
        )

    # Training callbacks
    es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1)
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1
    )

    # Train with progress bar
    with st.spinner("Training improved model... This may take a few minutes."):
        try:
            history = model.fit(
                X_train_s, y_train_s,
                validation_data=(X_test_s, y_test_s),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[es, mc, lr_schedule],
                verbose=1
            )

            st.success("✅ Training Complete with Improved Model!")

            # Store training history in session state
            st.session_state.training_history = history.history

            # Display training chart
            history_df = pd.DataFrame(history.history)
            st.line_chart(history_df[['loss','val_loss']])

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.warning("💡 Try reducing batch size or epochs, or restart the app and try again.")
            st.session_state.start_training = False
            st.stop()



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


    st.write(f"*MAE:* {mae:.2f}")
    st.write(f"*RMSE:* {rmse:.2f}")
    st.write(f"*R²:* {r2:.2f}")

    # Store results in session state
    st.session_state.model_results = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'samples_count': X.shape[0],
        'input_shape': X.shape[1:]
    }
    
    # Store evaluation data for visualization
    st.session_state.evaluation_results = {
        'y_test_inv': y_test_inv,
        'y_pred_inv': y_pred_inv
    }
    
    st.session_state.training_complete = True
    st.session_state.start_training = False  # Reset training flag

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
# Rename Items Section (Always Available)
# -------------------------
st.markdown("---")
st.markdown("## 🏷️ Rename Food Items")

# Check if model files exist
try:
    meta = joblib.load(META_PATH)
    model_exists = True
    current_mapping = meta.get('item_mapping', {f'item{i}': f'item{i}' for i in range(1, 6)})
except:
    model_exists = False
    current_mapping = {f'item{i}': f'item{i}' for i in range(1, 6)}

if model_exists:
    # Initialize session state for rename form
    if 'rename_form_key' not in st.session_state:
        st.session_state.rename_form_key = 0
    
    st.write("*Current Item Names:*")
    
    # Use session state to store form values
    if f'item_names_{st.session_state.rename_form_key}' not in st.session_state:
        st.session_state[f'item_names_{st.session_state.rename_form_key}'] = current_mapping.copy()
    
    # Create input fields outside of form first
    rename_dict = {}
    col1, col2 = st.columns(2)
    
    with col1:
        rename_dict['item1'] = st.text_input(
            "Item 1 Name:", 
            value=current_mapping.get('item1', 'item1'),
            key=f"item1_{st.session_state.rename_form_key}"
        )
        rename_dict['item2'] = st.text_input(
            "Item 2 Name:", 
            value=current_mapping.get('item2', 'item2'),
            key=f"item2_{st.session_state.rename_form_key}"
        )
        rename_dict['item3'] = st.text_input(
            "Item 3 Name:", 
            value=current_mapping.get('item3', 'item3'),
            key=f"item3_{st.session_state.rename_form_key}"
        )
    
    with col2:
        rename_dict['item4'] = st.text_input(
            "Item 4 Name:", 
            value=current_mapping.get('item4', 'item4'),
            key=f"item4_{st.session_state.rename_form_key}"
        )
        rename_dict['item5'] = st.text_input(
            "Item 5 Name:", 
            value=current_mapping.get('item5', 'item5'),
            key=f"item5_{st.session_state.rename_form_key}"
        )

    # Save button
    if st.button("💾 Save Item Names", key=f"save_btn_{st.session_state.rename_form_key}"):
        try:
            # Load and update metadata
            meta = joblib.load(META_PATH)
            meta['item_mapping'] = rename_dict
            meta['target_cols'] = ['Total_Orders'] + list(rename_dict.values())
            joblib.dump(meta, META_PATH)
            
            # Store in session state
            st.session_state['item_mapping'] = rename_dict
            st.session_state['target_cols'] = ['Total_Orders'] + list(rename_dict.values())
            
            # Increment form key to reset form with new values
            st.session_state.rename_form_key += 1
            
            st.success("✅ Item names saved successfully!")
            st.balloons()
            
            # Show updated mapping
            st.write("*Updated Item Names:*")
            for old_name, new_name in rename_dict.items():
                st.write(f"• {old_name} → *{new_name}*")
            
            # Force rerun to update the form
            st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error saving item names: {str(e)}")
else:
    st.warning("⚠️ Please train a model first before renaming items.")

# Only show training section if inside training flow
if uploaded_file is not None and st.session_state.get('start_training', False):
    
    # Continue with rest of training process that was moved...
    st.subheader("Step 5: Tomorrow Forecast Preview")
    
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_KEY_HERE":
        st.info("⚠️ Set your OpenWeather API key in code to enable live tomorrow prediction.")
    else:
        tomorrow_res = predict_future(model, input_scaler, output_scaler, df, OPENWEATHER_API_KEY)
        st.write("📌 Tomorrow's Prediction:")
        st.json(tomorrow_res)
        
        # Store tomorrow forecast in session state for persistent display
        st.session_state.tomorrow_forecast = tomorrow_res