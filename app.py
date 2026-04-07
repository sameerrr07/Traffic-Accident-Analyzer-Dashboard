import streamlit as st
import pandas as pd
import plotly.express as px
import json
from sklearn.ensemble import RandomForestRegressor
from streamlit_js_eval import streamlit_js_eval

# =========================
# 📱 AUTO MOBILE DETECTION
# =========================
screen_width = streamlit_js_eval(js_expressions='screen.width', key='WIDTH')

is_mobile = False
if screen_width is not None:
    is_mobile = screen_width < 768

st.set_page_config(
    page_title="Traffic Analyzer",
    layout="wide",
    initial_sidebar_state="auto"
)
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 1rem; }
.stMetric {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}
[data-testid="stSidebar"] { background-color: #111827; }
h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
@media (max-width: 768px) {
    .stMetric {
        font-size: 12px !important;
        padding: 10px !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("🚦 India Traffic & Accident Dashboard")
st.markdown("## 📊 Overview")

# =========================
# 📂 LOAD DATA
# =========================
with st.spinner("Loading Dashboard..."):
    df = pd.read_csv("data/ADSI_Table_1A.2.csv")

df.columns = df.columns.str.strip()
state_col = df.columns[1]
df = df[df[state_col] != "Total"]

df["Death Rate"] = df["Total Traffic Accidents - Died"] / df["Total Traffic Accidents - Cases"]

# =========================
# 🔎 FILTER + COMPARE
# =========================
st.sidebar.header("🔎 Filters")
st.sidebar.markdown("## ⚖️ Compare States")

state1 = st.sidebar.selectbox("State 1", df[state_col].unique())
state2 = st.sidebar.selectbox("State 2", df[state_col].unique())

if state1 == state2:
    st.sidebar.warning("⚠️ Please select different states")

selected_states = st.sidebar.multiselect(
    "Select States",
    options=df[state_col].unique(),
    default=df[state_col].unique()
)

# =========================
# 📊 FILTERED DATA
# =========================

filtered_df = df[df[state_col].isin(selected_states)].copy()

# =========================
# 🤖 ML MODEL TRAINING
# =========================
model_df = df.copy()

# state encoding
model_df["State_Code"] = model_df[state_col].astype("category").cat.codes

X = model_df[["State_Code"]]
y = model_df["Total Traffic Accidents - Cases"]

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)

# Compare (correct order)
compare_df = filtered_df[filtered_df[state_col].isin([state1, state2])]

# =========================
# 🧠 RISK CLASSIFICATION
# =========================
def classify_risk(value):
    if value > 50000:
        return "High"
    elif value > 20000:
        return "Medium"
    else:
        return "Low"

filtered_df["Risk Level"] = filtered_df["Total Traffic Accidents - Cases"].apply(classify_risk)

# =========================
# 📊 KPI
# =========================
if is_mobile:
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
else:
    col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accidents", int(filtered_df["Total Traffic Accidents - Cases"].sum()))
col2.metric("Total Deaths", int(filtered_df["Total Traffic Accidents - Died"].sum()))
if not filtered_df.empty:
    col3.metric("Most Affected State", filtered_df.loc[
        filtered_df["Total Traffic Accidents - Cases"].idxmax(), state_col
    ])
else:
    col3.metric("Most Affected State", "N/A")
col4.metric("Avg Death Rate", f"{filtered_df['Death Rate'].mean():.2%}")

# =========================
# 📈 GRAPHS
# =========================
st.markdown("## 📈 Analysis")

top_states = filtered_df.sort_values(
    by="Total Traffic Accidents - Cases",
    ascending=False
).head(10)

fig = px.bar(
    top_states,
    x=state_col,
    y="Total Traffic Accidents - Cases",
    color="Total Traffic Accidents - Cases",
    title="Top 10 Accident States"
)

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

danger_states = filtered_df.sort_values(
    by="Death Rate",
    ascending=False
).head(10)

fig2 = px.bar(
    danger_states,
    x=state_col,
    y="Death Rate",
    color="Death Rate",
    title="Most Dangerous States"
)

fig2.update_layout(template="plotly_dark")
st.plotly_chart(fig2, use_container_width=True)

# =========================
# ⚖️ COMPARISON
# =========================
st.markdown("## ⚖️ State Comparison")

if not compare_df.empty:
    st.bar_chart(compare_df.set_index(state_col)["Total Traffic Accidents - Cases"])
else:
    st.warning("⚠️ Selected states not in filtered data")

# =========================
# 🗺️ MAP
# =========================
# 🗺️ ADVANCED MAP SECTION
# =========================
with open("data/india.geojson", encoding="utf-8") as f:
    geojson = json.load(f)

# key detect
sample_props = geojson["features"][0]["properties"]
possible_keys = ["ST_NM", "state", "NAME_1", "name", "STATE"]

geo_key = None
for key in possible_keys:
    if key in sample_props:
        geo_key = key
        break

if geo_key is None:
    st.error("❌ State name key not found")
    st.stop()

map_df = filtered_df.copy()
map_df[state_col] = map_df[state_col].str.strip()

# state name fix
map_df[state_col] = map_df[state_col].replace({
    "Delhi": "NCT of Delhi",
    "Odisha": "Orissa",
    "Uttarakhand": "Uttaranchal",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "Andaman & Nicobar": "Andaman and Nicobar Islands",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli",
    "Daman & Diu": "Daman and Diu"
})

geo_states = [f["properties"][geo_key] for f in geojson["features"]]
map_df = map_df[map_df[state_col].isin(geo_states)]

# =========================
# 🎯 MAP WITH RICH DATA
# =========================
fig_map = px.choropleth(
    map_df,
    geojson=geojson,
    featureidkey=f"properties.{geo_key}",
    locations=state_col,
    color="Total Traffic Accidents - Cases",
    color_continuous_scale="Turbo",
    title="🗺️ Smart Traffic Heatmap",
    hover_name=state_col,
    hover_data={
        "Total Traffic Accidents - Cases": True,
        "Death Rate": True,
        "Risk Level": True
    }
)

# smooth UI
fig_map.update_geos(
    visible=False,
    showcountries=False,
    showcoastlines=False,
    fitbounds="locations"
)

fig_map.update_layout(
    template="plotly_dark",
    height=700,
    margin={"r":0,"t":50,"l":0,"b":0},
    coloraxis_colorbar=dict(title="Accidents")
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# 📍 SELECT STATE DETAILS
# =========================
st.markdown("## 📍 State Details Viewer")

selected_state = st.selectbox(
    "Select a state to view detailed info",
    map_df[state_col].unique()
)

state_data = map_df[map_df[state_col] == selected_state]

if not state_data.empty:
    st.success(f"📊 Details for {selected_state}")
    
    st.write("🚗 Total Accidents:", int(state_data["Total Traffic Accidents - Cases"].values[0]))
    st.write("💀 Death Rate:", f"{state_data['Death Rate'].values[0]:.2%}")
    st.write("⚠️ Risk Level:", state_data["Risk Level"].values[0])
# =========================

# =========================
# 🚨 RISK TABLE
# =========================
st.markdown("## 🚨 Risk Classification")

st.dataframe(
    filtered_df[[state_col, "Total Traffic Accidents - Cases", "Risk Level"]]
)

# =========================
# 💡 INSIGHTS
# =========================
st.markdown("## 💡 Key Insights")
if not filtered_df.empty:
    growth_state = filtered_df.sort_values(
        by="Total Traffic Accidents - Cases",
        ascending=False
    ).iloc[0][state_col]

    low_state = filtered_df.sort_values(
        by="Total Traffic Accidents - Cases",
        ascending=True
    ).iloc[0][state_col]

    st.success(f"📈 {growth_state} needs urgent traffic control measures")
    st.info(f"✅ {low_state} is comparatively safer state")
else:
    st.warning("⚠️ No data available")
# =========================
# 📥 DOWNLOAD
# =========================
st.download_button(
    "📥 Download Report",
    data=filtered_df.to_csv(index=False),
    file_name="traffic_report.csv"
)