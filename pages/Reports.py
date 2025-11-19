import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import joblib

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
st.set_page_config(page_title="📊 Reports - Food Demand Forecast", layout="wide")

st.title("📊 Reports Dashboard")
st.markdown("Get insights on sales, demand patterns, and model performance.")

# -------------------------
# Load Data & Metadata (Updated)
# -------------------------
import os
import joblib

DATA_PATH = "restaurant_sales_updated_items.csv"
META_PATH = "model_meta.joblib"

@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def load_meta(path=META_PATH):
    # Load metadata safely
    if os.path.exists(path):
        try:
            meta = joblib.load(path)
        except Exception as e:
            st.warning(f"⚠️ Failed to load meta file: {e}. Using defaults.")
            meta = {}
    else:
        meta = {}

    # Ensure defaults
    default_map = {
        'item1': 'item1', 'item2': 'item2',
        'item3': 'item3', 'item4': 'item4', 'item5': 'item5'
    }
    meta.setdefault('item_mapping', default_map)
    meta.setdefault('target_cols', ['Total_Orders'] + list(default_map.values()))

    # Prefer session_state overrides if present
    if 'item_mapping' in st.session_state:
        ss_map = st.session_state.get('item_mapping', {})
        if isinstance(ss_map, dict) and ss_map:
            meta['item_mapping'] = ss_map
            meta['target_cols'] = st.session_state.get('target_cols', meta['target_cols'])

    return meta

# Load dataset and metadata
df = load_data()
meta = load_meta()

# Extract mappings
item_mapping = meta['item_mapping']
target_cols = meta['target_cols']

# Only rename columns that exist in the dataframe
rename_map = {k: v for k, v in item_mapping.items() if k in df.columns}
df_renamed = df.rename(columns=rename_map)

# Keep session_state updated so other pages see correct names
st.session_state['item_mapping'] = item_mapping
st.session_state['target_cols'] = target_cols

# -------------------------
# Section 1: Sales Trend Analysis
# -------------------------
st.subheader("📈 Sales Trend Analysis")

option = st.selectbox(
    "Select Metric",
    ["Total Orders"] + list(rename_map.values())
)
time_filter = st.selectbox("Select Time Range", ["Last 6 Months", "Last 3 Months", "Last 1 Month", "Last Week"])

today = df_renamed['Date'].max()
if time_filter == "Last 6 Months":
    start_date = today - pd.DateOffset(months=6)
elif time_filter == "Last 3 Months":
    start_date = today - pd.DateOffset(months=3)
elif time_filter == "Last 1 Month":
    start_date = today - pd.DateOffset(months=1)
else:
    start_date = today - pd.DateOffset(weeks=1)

df_filtered = df_renamed[df_renamed['Date'] >= start_date]

fig, ax = plt.subplots(figsize=(10,5))
if option == "Total Orders":
    ax.plot(df_filtered['Date'], df_filtered['Total_Orders'], label="Total Orders", color="orange")
else:
    ax.plot(df_filtered['Date'], df_filtered[option], label=option, color="green")

ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.set_title(f"{option} Trend ({time_filter})")
ax.legend()
st.pyplot(fig)

# -------------------------
# Section 2: Per-Item Sales Breakdown
# -------------------------
st.subheader("🥗 Per-Item Sales Breakdown")

breakdown_filter = st.selectbox(
    "Select Time Range",
    ["Last 6 Months", "Last 3 Months", "Last 1 Month", "Last Week"],
    key="item_breakdown_filter"
)

if breakdown_filter == "Last 6 Months":
    start_date = today - pd.DateOffset(months=6)
elif breakdown_filter == "Last 3 Months":
    start_date = today - pd.DateOffset(months=3)
elif breakdown_filter == "Last 1 Month":
    start_date = today - pd.DateOffset(months=1)
else:
    start_date = today - pd.DateOffset(weeks=1)

period_df = df_renamed[df_renamed['Date'] >= start_date]

# Item sums
item_cols = list(rename_map.values())
item_sums = period_df[item_cols].sum()

plt.style.use("dark_background")
fig_pie, ax_pie = plt.subplots(figsize=(3,3))
ax_pie.pie(
    item_sums,
    labels=item_sums.index,
    autopct="%1.1f%%",
    startangle=90,
    textprops={'color':"w"}
)
ax_pie.set_title(f"Item-wise Sales Share ({breakdown_filter})", color="w")
st.pyplot(fig_pie)

# -------------------------
# Section 3: Top Performers
# -------------------------
st.subheader("🏆 Top Performing Items")

top_time_filter = st.selectbox(
    "Select Time Range for Top Performers",
    ["Last 6 Months", "Last 3 Months", "Last 1 Month", "Last Week"],
    key="top_perf_filter"
)

if top_time_filter == "Last 6 Months":
    start_date = today - pd.DateOffset(months=6)
elif top_time_filter == "Last 3 Months":
    start_date = today - pd.DateOffset(months=3)
elif top_time_filter == "Last 1 Month":
    start_date = today - pd.DateOffset(months=1)
else:
    start_date = today - pd.DateOffset(weeks=1)

df_top = df_renamed[df_renamed['Date'] >= start_date]

item_totals = df_top[list(rename_map.values())].sum().sort_values(ascending=False)

st.bar_chart(item_totals.head(5))
st.write(item_totals.head(10))

# -------------------------
# Section 4: Model Accuracy (Actual vs Predicted)
# -------------------------
st.subheader("🎯 Model Accuracy (Actual vs Predicted)")

try:
    pred_df = pd.read_csv("predictions.csv")
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])

    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(pred_df['Date'], pred_df['Actual'], label="Actual", color="blue")
    ax2.plot(pred_df['Date'], pred_df['Predicted'], label="Predicted", color="red", linestyle="--")
    ax2.set_title("Actual vs Predicted Orders")
    ax2.legend()
    st.pyplot(fig2)
except:
    st.info("⚠️ Predictions file not found. Train model first to see accuracy report.")
