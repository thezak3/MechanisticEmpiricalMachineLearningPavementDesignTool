import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="TxDOT Pavement Design Tool",
    page_icon="🛣️",
    layout="wide"
)

# Title and description
st.title("🛣️ TxDOT Pavement Fatigue Cracking Design Tool")
st.markdown("**Predict pavement fatigue cracking using machine learning models**")
st.markdown("---")

# Load models and scaler
@st.cache_resource
def load_models():
    with open('xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('lightgbm_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return xgb_model, lgb_model, rf_model, scaler

# Load models
try:
    xgb_model, lgb_model, rf_model, scaler = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Binder grade and mix type data (from your table)
binder_data = {
    ('64-22', 'Type B'): (6.4359E-06, 3.8374),
    ('70-22', 'Type B'): (6.8551E-06, 3.8201),
    ('76-22', 'Type B'): (7.317E-06, 3.8023),
    ('64-28', 'Type B'): (3.1557E-06, 4.0323),
    ('70-28', 'Type B'): (3.7800E-06, 3.9828),
    ('64-22', 'Type C'): (5.2041E-06, 3.8948),
    ('70-22', 'Type C'): (5.5095E-06, 3.8792),
    ('76-22', 'Type C'): (5.8430E-06, 3.8630),
    ('64-28', 'Type C'): (2.8039E-06, 4.0645),
    ('70-28', 'Type C'): (3.3231E-06, 4.0179),
    ('64-22', 'Type D'): (4.2081E-06, 3.9531),
    ('70-22', 'Type D'): (4.4280E-06, 3.9391),
    ('76-22', 'Type D'): (4.6659E-06, 3.9248),
    ('64-28', 'Type D'): (2.4914E-06, 4.0969),
    ('70-28', 'Type D'): (2.9215E-06, 4.0532),
    ('64-22', 'Superpave B'): (6.0544E-06, 3.8541),
    ('70-22', 'Superpave B'): (6.4359E-06, 3.8374),
    ('76-22', 'Superpave B'): (6.8551E-06, 3.8201),
    ('64-28', 'Superpave B'): (3.0241E-06, 4.0440),
    ('70-28', 'Superpave B'): (3.6074E-06, 3.9956),
    ('64-22', 'Superpave C'): (4.9238E-06, 3.9100),
    ('70-22', 'Superpave C'): (5.2041E-06, 3.8948),
    ('76-22', 'Superpave C'): (5.5095E-06, 3.8792),
    ('64-28', 'Superpave C'): (2.6934E-06, 4.0755),
    ('70-28', 'Superpave C'): (3.1804E-06, 4.0299),
    ('64-22', 'Superpave D'): (4.0044E-06, 3.9667),
    ('70-22', 'Superpave D'): (4.2081E-06, 3.9531),
    ('76-22', 'Superpave D'): (4.4280E-06, 3.9391),
    ('64-28', 'Superpave D'): (2.3989E-06, 4.1073),
    ('70-28', 'Superpave D'): (2.8039E-06, 4.0645),
    ('76-22', 'SMA-C'): (9.2769E-08, 4.9996),
    ('76-22', 'SMA-D'): (8.1315E-08, 5.0358),
    ('76-22', 'SMA-F'): (6.0576E-08, 5.1166),
    ('70-28', 'SMA-C'): (9.2769E-08, 4.9996),
    ('70-28', 'SMA-D'): (8.1315E-08, 5.0358),
}

# Sidebar - Input Parameters
st.sidebar.header("📋 Design Input Parameters")

# Design Life Selection
design_life = st.sidebar.selectbox(
    "Design Life (years)",
    [5, 10, 15, 20],
    index=3  # Default to 20 years
)

st.sidebar.markdown("---")
st.sidebar.subheader("Mix Design")

# Binder Grade and Mix Type Selection
pg_options = sorted(list(set([key[0] for key in binder_data.keys()])))
selected_pg = st.sidebar.selectbox("Performance Grade (PG)", pg_options, index=0)

# Filter mix types based on selected PG
available_mix_types = sorted(list(set([key[1] for key in binder_data.keys() if key[0] == selected_pg])))
selected_mix = st.sidebar.selectbox("Mix Type", available_mix_types, index=0)

# Get A and n values
A_raw, n = binder_data[(selected_pg, selected_mix)]
A = A_raw * 1e6  # Scale A as in training

# RAP Percent
rap_percent = st.sidebar.slider("RAP Content (%)", 0, 50, 25, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Pavement Structure")

# AC Layer
ac_thickness = st.sidebar.number_input("AC Thickness (inches)", 4.0, 12.0, 8.0, 0.5)
ac_modulus = st.sidebar.number_input("AC Modulus (ksi)", 200, 3000, 1200, 50)

# Base Layer
base_thickness = st.sidebar.number_input("Base Thickness (inches)", 4.0, 24.0, 12.0, 1.0)
base_modulus = st.sidebar.number_input("Base Modulus (psi)", 10000, 50000, 25000, 1000)

# Subgrade
subgrade_modulus = st.sidebar.number_input("Subgrade Modulus (psi)", 3000, 20000, 8000, 500)

st.sidebar.markdown("---")
st.sidebar.subheader("Traffic Loading")

# Traffic level
traffic_level = st.sidebar.selectbox(
    "Traffic Level",
    ["Light", "Medium", "Heavy", "Custom"],
    index=1
)

if traffic_level == "Light":
    total_esals = 500000
elif traffic_level == "Medium":
    total_esals = 2000000
elif traffic_level == "Heavy":
    total_esals = 5000000
else:  # Custom
    total_esals = st.sidebar.number_input(
        "Total Design ESALs",
        100000, 10000000, 2000000, 100000
    )

# Calculate monthly ESALs based on design life
months = design_life * 12
cumulative_monthly_esals = total_esals  # Total ESALs at end of design life

st.sidebar.markdown("---")

# Predict button
if st.sidebar.button("🔍 Calculate Cracking Prediction", type="primary"):
    
    # Create feature array for the selected design life
    pavement_age_months = months
    
    # Base features
    features_base = {
        'Pavement_Age_Months': pavement_age_months,
        'Cumulative_Monthly_ESALs': cumulative_monthly_esals,
        'AC_Thickness': ac_thickness,
        'RAP_Percent': rap_percent,
        'A': A,
        'n': n,
        'AC_Modulus_ksi': ac_modulus,
        'Base_Thickness': base_thickness,
        'Base_Modulus': base_modulus,
        'Subgrade_Modulus': subgrade_modulus
    }
    
    # Create DataFrame
    df_input = pd.DataFrame([features_base])
    
    # Add polynomial and interaction features
    df_input['Age_Squared'] = df_input['Pavement_Age_Months'] ** 2
    df_input['ESALs_Squared'] = df_input['Cumulative_Monthly_ESALs'] ** 2
    df_input['Age_x_ESALs'] = df_input['Pavement_Age_Months'] * df_input['Cumulative_Monthly_ESALs']
    df_input['AC_Thickness_x_Modulus'] = df_input['AC_Thickness'] * df_input['AC_Modulus_ksi']
    df_input['Base_Thickness_x_Modulus'] = df_input['Base_Thickness'] * df_input['Base_Modulus']
    
    # Add Paris Law features
    df_input['Paris_ESALs'] = df_input['A'] * (df_input['Cumulative_Monthly_ESALs'] ** df_input['n'])
    df_input['Paris_Age'] = df_input['A'] * (df_input['Pavement_Age_Months'] ** df_input['n'])
    df_input['Paris_Combined'] = df_input['A'] * ((df_input['Cumulative_Monthly_ESALs'] + df_input['Pavement_Age_Months']) ** df_input['n'])
    df_input['Paris_Modulus_ESALs'] = (df_input['A'] / (df_input['AC_Modulus_ksi'] + 1)) * (df_input['Cumulative_Monthly_ESALs'] ** df_input['n'])
    
    # Normalize features
    X_scaled = scaler.transform(df_input)
    
    # Make predictions (on sqrt-transformed scale)
    pred_xgb_sqrt = xgb_model.predict(X_scaled)[0]
    pred_lgb_sqrt = lgb_model.predict(X_scaled)[0]
    pred_rf_sqrt = rf_model.predict(X_scaled)[0]
    
    # Reverse sqrt transformation and clip
    pred_xgb = np.clip(pred_xgb_sqrt ** 2, 0, 100)
    pred_lgb = np.clip(pred_lgb_sqrt ** 2, 0, 100)
    pred_rf = np.clip(pred_rf_sqrt ** 2, 0, 100)
    
    # Average prediction
    avg_prediction = (pred_xgb + pred_lgb + pred_rf) / 3
    
    # Store in session state
    st.session_state['predictions'] = {
        'xgb': pred_xgb,
        'lgb': pred_lgb,
        'rf': pred_rf,
        'avg': avg_prediction,
        'design_life': design_life
    }

# Display results if predictions exist
if 'predictions' in st.session_state:
    preds = st.session_state['predictions']
    avg_pred = preds['avg']
    
    # Determine status
    if avg_pred < 15:
        status = "Good"
        color = "green"
        emoji = "✅"
    elif avg_pred <= 30:
        status = "Acceptable"
        color = "orange"
        emoji = "⚠️"
    else:
        status = "Early Failure"
        color = "red"
        emoji = "❌"
    
    # Main results display
    st.markdown("## 📊 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"Average Predicted Cracking at {preds['design_life']} years",
            value=f"{avg_pred:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Design Status",
            value=status,
            delta=None
        )
    
    with col3:
        st.markdown(f"### {emoji} {status}")
        if status == "Good":
            st.success("Design meets performance criteria")
        elif status == "Acceptable":
            st.warning("Design is acceptable but monitor closely")
        else:
            st.error("Design may experience early failure")
    
    st.markdown("---")
    
    # Individual model predictions
    st.markdown("### 🤖 Individual Model Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("XGBoost", f"{preds['xgb']:.2f}%")
    with col2:
        st.metric("LightGBM", f"{preds['lgb']:.2f}%")
    with col3:
        st.metric("Random Forest", f"{preds['rf']:.2f}%")
    
    # Gauge chart
    st.markdown("### 📈 Performance Gauge")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_pred,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fatigue Cracking (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 30], 'color': "yellow"},
                {'range': [30, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Design criteria reference
    st.markdown("---")
    st.markdown("### 📋 Design Criteria")
    
    criteria_df = pd.DataFrame({
        'Status': ['Good', 'Acceptable', 'Early Failure'],
        'Cracking Range': ['< 15%', '15% - 30%', '> 30%'],
        'Recommendation': [
            'Design meets long-term performance goals',
            'Design is acceptable; consider monitoring',
            'Design may require modification or increased maintenance'
        ]
    })
    
    st.table(criteria_df)

else:
    # Initial instructions
    st.info("👈 Enter your design parameters in the sidebar and click **Calculate** to see predictions")
    
    st.markdown("### 📖 How to Use This Tool")
    st.markdown("""
    1. **Select Design Life** - Choose 5, 10, 15, or 20 years
    2. **Choose Mix Design** - Select Performance Grade (PG) and Mix Type
    3. **Enter Pavement Structure** - Input layer thicknesses and moduli
    4. **Specify Traffic** - Select traffic level or enter custom ESALs
    5. **Click Calculate** - View predicted fatigue cracking percentage
    
    The tool uses three machine learning models (XGBoost, LightGBM, Random Forest) trained on TxDOT pavement data to predict fatigue cracking.
    """)
    
    st.markdown("### ⚙️ Model Information")
    st.markdown("""
    - **Models**: XGBoost, LightGBM, Random Forest ensemble
    - **Training Data**: 260 pavement cases, 62,400 observations
    - **Key Features**: Paris Law parameters (A, n), pavement structure, traffic loading
    - **Prediction**: Average of three model outputs for robust results
    """)

# Footer
st.markdown("---")
st.markdown("*TxDOT Pavement Fatigue Cracking Design Tool | Developed with Machine Learning*")