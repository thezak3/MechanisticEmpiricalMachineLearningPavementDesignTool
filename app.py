import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Pavement Design Tool",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modal overlay
st.markdown("""
<style>
.modal-overlay {
    display: none;
    position: fixed;
    z-index: 9999;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    overflow: auto;
    background-color: rgba(0,0,0,0.6);
}

.modal-content {
    background-color: #fefefe;
    margin: 5% auto;
    padding: 30px;
    border: 1px solid #888;
    border-radius: 10px;
    width: 80%;
    max-width: 800px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

.close-modal {
    color: #aaa;
    float: right;
    font-size: 28px;
    font-weight: bold;
    cursor: pointer;
    line-height: 20px;
}

.close-modal:hover,
.close-modal:focus {
    color: #000;
}

.info-button {
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    padding: 5px;
    color: #0066cc;
}

.info-button:hover {
    color: #0052a3;
}
</style>

<script>
function showModal() {
    document.getElementById('infoModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('infoModal').style.display = 'none';
}

// Close modal if user clicks outside of it
window.onclick = function(event) {
    var modal = document.getElementById('infoModal');
    if (event.target == modal) {
        modal.style.display = 'none';
    }
}
</script>
""", unsafe_allow_html=True)

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

# Binder grade and mix type data
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

# AC Modulus lookup table (average values from training data by PG grade and Mix Type)
ac_modulus_lookup = {
    ('64-22', 'Type B'): 1389.46,
    ('76-22', 'Type B'): 1652.78,
    ('64-28', 'Type B'): 1244.88,
    ('64-28', 'Type C'): 1152.44,
    ('70-22', 'Type D'): 1346.05,
    ('76-22', 'Type D'): 1449.94,
    ('64-28', 'Type D'): 1101.56,
    ('70-22', 'Superpave B'): 1389.46,
    ('70-28', 'Superpave C'): 1244.88,
    ('76-22', 'Superpave D'): 1346.05,
    ('70-28', 'Superpave D'): 1152.44,
}

# Default AC Modulus for combinations not in training data
DEFAULT_AC_MODULUS = 837

# Training data ranges for validation
VALID_RANGES = {
    'AC_Thickness': (4.0, 7.0),
    'Base_Thickness': (8.0, 24.0),
    'Base_Modulus': (36.5, 250),
    'Subgrade_Modulus': (5.0, 20.0),
    'RAP_Percent': (0, 30)
}

# Prediction function
def predict_cracking(inputs_dict, scaler, xgb_model, lgb_model, rf_model):
    """Make cracking prediction for given inputs"""
    df_input = pd.DataFrame([inputs_dict])
    
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
    
    # Scale features
    X_scaled = scaler.transform(df_input)
    
    # Predict
    pred_xgb = np.clip(xgb_model.predict(X_scaled)[0] ** 2, 0, 100)
    pred_lgb = np.clip(lgb_model.predict(X_scaled)[0] ** 2, 0, 100)
    pred_rf = np.clip(rf_model.predict(X_scaled)[0] ** 2, 0, 100)
    
    return pred_xgb, pred_lgb, pred_rf

# Check if value is out of range
def check_out_of_range(param_name, value):
    if param_name in VALID_RANGES:
        min_val, max_val = VALID_RANGES[param_name]
        return value < min_val or value > max_val
    return False

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'instructions'

# Instructions Page
if st.session_state.page == 'instructions':
    st.title("Pavement Fatigue Cracking Design Tool")
    st.markdown("## Instructions")
    
    st.markdown("""
    ### Purpose
    This tool predicts fatigue cracking in flexible pavements using machine learning models trained on pavement performance data.
    
    ### How to Use
    
    **1. Design Input Tab**
    - Select your pavement design parameters:
        - Design life (5, 10, 15, or 20 years)
        - Mix design (Performance Grade and Mix Type)
        - Pavement structure (layer thicknesses and moduli)
        - Traffic loading
    - Click "Calculate Prediction" to see results
    
    **2. Design Comparison Tab**
    - Compare three design alternatives side-by-side:
        - Your base design
        - A "beefed up" design (thicker AC layer)
        - A "thinned" design (thinner AC layer)
    - View cracking progression over time for all three designs
    
    **3. Understanding Results**
    - **Good** (Green): Cracking < 15% - Design meets performance criteria
    - **Acceptable** (Yellow): Cracking 15-30% - Design is acceptable but monitor closely
    - **Early Failure** (Red): Cracking > 30% - Design may require modification
    
    **4. Cracking Progression**
    - Graph shows predicted cracking from year 0 to your selected design life
    - Helps identify when cracking accelerates and plan maintenance
    
    **5. Sensitivity Analysis**
    - Shows how changing key parameters affects cracking
    - Helps optimize your design
    
    ### Models
    - **XGBoost, LightGBM, Random Forest** - Three machine learning models
    - Predictions are averaged for robust results
    - Trained on 260 pavement cases with 62,400 observations
    
    ### Limitations
    - Predictions are most reliable within the training data ranges
    - Warnings appear if inputs are outside typical values
    - This tool aids engineering judgment but does not replace it
    
    ### Data Ranges
    - AC Thickness: 4.0 - 7.0 inches
    - Base Thickness: 8.0 - 24.0 inches
    - Base Modulus: 36.5 - 250 ksi
    - Subgrade Modulus: 5.0 - 20.0 ksi
    - RAP Content: 0 - 30%
    """)
    
    if st.button("I Understand - Proceed to Tool", type="primary", key="proceed_button"):
        st.session_state.page = 'design_input'
        st.rerun()

# Main App Pages
else:
    # Load models
    try:
        xgb_model, lgb_model, rf_model, scaler = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Design Input", "Design Comparison Tool"])
    
    # ============================================================================
    # TAB 1: DESIGN INPUT
    # ============================================================================
    with tab1:
        st.title("Pavement Fatigue Cracking Design Tool")
        
        # Input Section
        st.markdown("## Design Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Design Life & Mix")
            design_life = st.selectbox("Design Life (years)", [5, 10, 15, 20], index=3)
            
            pg_options = sorted(list(set([key[0] for key in binder_data.keys()])))
            selected_pg = st.selectbox("Performance Grade (PG)", pg_options, index=0)
            
            available_mix_types = sorted(list(set([key[1] for key in binder_data.keys() if key[0] == selected_pg])))
            selected_mix = st.selectbox("Mix Type", available_mix_types, index=0)
            
            A_raw, n = binder_data[(selected_pg, selected_mix)]
            A_scaled = A_raw * 1e6
            
            # Get AC_Modulus from lookup table
            ac_modulus = ac_modulus_lookup.get((selected_pg, selected_mix), DEFAULT_AC_MODULUS)
            
            # Show AC Modulus (auto-calculated, not editable)
            st.info(f"AC Modulus: {ac_modulus:.0f} ksi (auto-calculated from mix selection)")
            
            rap_percent = st.slider("RAP Content (%)", 0, 50, 15, 5)
        
        with col2:
            st.markdown("### Pavement Structure")
            ac_thickness = st.selectbox(
                "AC Thickness (inches)", 
                [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 
                index=3
            )
            base_thickness = st.selectbox(
                "Base Thickness (inches)", 
                [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0], 
                index=4
            )
            base_modulus = st.number_input("Base Modulus (ksi)", min_value=30, max_value=300, value=37, step=5)
            subgrade_modulus = st.number_input("Subgrade Modulus (ksi)", min_value=5, max_value=20, value=10, step=5)
        
        with col3:
            st.markdown("### Traffic Loading")
            traffic_level = st.selectbox("Traffic Level", ["Light", "Medium", "Heavy", "Custom"], index=1)
            
            if traffic_level == "Light":
                total_esals = 500000
                st.info(f"Total ESALs: {total_esals:,}")
            elif traffic_level == "Medium":
                total_esals = 2000000
                st.info(f"Total ESALs: {total_esals:,}")
            elif traffic_level == "Heavy":
                total_esals = 5000000
                st.info(f"Total ESALs: {total_esals:,}")
            else:
                total_esals = st.number_input("Total Design ESALs", 100000, 10000000, 2000000, 100000)
        
        # Check for out-of-range values
        warnings = []
        if check_out_of_range('AC_Thickness', ac_thickness):
            warnings.append(f"AC Thickness ({ac_thickness} in) is outside typical range {VALID_RANGES['AC_Thickness']}")
        if check_out_of_range('Base_Thickness', base_thickness):
            warnings.append(f"Base Thickness ({base_thickness} in) is outside typical range {VALID_RANGES['Base_Thickness']}")
        if check_out_of_range('Base_Modulus', base_modulus):
            warnings.append(f"Base Modulus ({base_modulus} ksi) is outside typical range {VALID_RANGES['Base_Modulus']}")
        if check_out_of_range('Subgrade_Modulus', subgrade_modulus):
            warnings.append(f"Subgrade Modulus ({subgrade_modulus} ksi) is outside typical range {VALID_RANGES['Subgrade_Modulus']}")
        if check_out_of_range('RAP_Percent', rap_percent):
            warnings.append(f"RAP Content ({rap_percent}%) is outside typical range {VALID_RANGES['RAP_Percent']}")
        
        if warnings:
            st.warning("Out-of-Range Inputs Detected:\n" + "\n".join([f"- {w}" for w in warnings]) + 
                      "\n\nPredictions may be less reliable for values outside training data ranges.")
        
        # Calculate button
        if st.button("Calculate Prediction", type="primary", key="calc_button"):
            months = design_life * 12
            cumulative_monthly_esals = (total_esals / 240) * months
            
            # Prepare inputs
            inputs_dict = {
                'Pavement_Age_Months': months,
                'Cumulative_Monthly_ESALs': cumulative_monthly_esals,
                'AC_Thickness': ac_thickness,
                'RAP_Percent': rap_percent,
                'A': A_scaled,
                'n': n,
                'AC_Modulus_ksi': ac_modulus,
                'Base_Thickness': base_thickness,
                'Base_Modulus': base_modulus,
                'Subgrade_Modulus': subgrade_modulus
            }
            
            # Make prediction
            pred_xgb, pred_lgb, pred_rf = predict_cracking(inputs_dict, scaler, xgb_model, lgb_model, rf_model)
            avg_pred = (pred_xgb + pred_lgb + pred_rf) / 3
            
            # Calculate model confidence
            model_spread = max(pred_xgb, pred_lgb, pred_rf) - min(pred_xgb, pred_lgb, pred_rf)
            if model_spread < 5:
                confidence = "High"
            elif model_spread < 10:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Store in session
            st.session_state['predictions'] = {
                'xgb': pred_xgb,
                'lgb': pred_lgb,
                'rf': pred_rf,
                'avg': avg_pred,
                'design_life': design_life,
                'confidence': confidence,
                'spread': model_spread
            }
            
            # Generate cracking progression
            years = list(range(design_life + 1))
            progression = []
            
            for year in years:
                year_months = year * 12
                year_esals = (total_esals / 240) * year_months if year > 0 else 0
                
                year_inputs = inputs_dict.copy()
                year_inputs['Pavement_Age_Months'] = year_months
                year_inputs['Cumulative_Monthly_ESALs'] = year_esals
                
                y_xgb, y_lgb, y_rf = predict_cracking(year_inputs, scaler, xgb_model, lgb_model, rf_model)
                y_avg = (y_xgb + y_lgb + y_rf) / 3
                progression.append(y_avg)
            
            st.session_state['progression'] = {'years': years, 'cracking': progression}
            st.session_state['base_inputs'] = inputs_dict
            st.session_state['total_esals'] = total_esals
            st.session_state['selected_pg'] = selected_pg
        
        # Display results
        if 'predictions' in st.session_state:
            preds = st.session_state['predictions']
            avg_pred = preds['avg']
            
            # Determine status
            if avg_pred < 15:
                status = "Good"
            elif avg_pred <= 30:
                status = "Acceptable"
            else:
                status = "Early Failure"
            
            st.markdown("---")
            
            # Header with info button
            col_header1, col_header2 = st.columns([0.95, 0.05])
            with col_header1:
                st.markdown("## Prediction Results")
            with col_header2:
                st.markdown("""
                <button class="info-button" onclick="showModal()">ℹ️</button>
                """, unsafe_allow_html=True)
            
            # Modal overlay
            st.markdown(f"""
            <div id="infoModal" class="modal-overlay">
                <div class="modal-content">
                    <span class="close-modal" onclick="closeModal()">&times;</span>
                    <h2>Model Confidence Information</h2>
                    <p><strong>Confidence Level:</strong> {preds['confidence']}</p>
                    <p><strong>Model Spread:</strong> {preds['spread']:.2f}%</p>
                    <hr>
                    <h3>What does this mean?</h3>
                    <ul>
                        <li><strong>High Confidence</strong> (spread &lt; 5%): Models agree closely, prediction is highly reliable</li>
                        <li><strong>Medium Confidence</strong> (spread 5-10%): Moderate agreement, prediction is reasonably reliable</li>
                        <li><strong>Low Confidence</strong> (spread &gt; 10%): Models disagree, use caution with prediction</li>
                    </ul>
                    <p><em>Model spread = difference between highest and lowest model prediction</em></p>
                    <br>
                    <button onclick="closeModal()" style="padding: 10px 20px; background-color: #0066cc; color: white; border: none; border-radius: 5px; cursor: pointer;">Close</button>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label=f"Predicted Cracking at {preds['design_life']} years", value=f"{avg_pred:.2f}%")
            
            with col2:
                st.metric(label="Design Status", value=status)
            
            with col3:
                st.metric(label="Model Agreement", value=preds['confidence'])
            
            # Individual predictions
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("XGBoost", f"{preds['xgb']:.2f}%")
            with col2:
                st.metric("LightGBM", f"{preds['lgb']:.2f}%")
            with col3:
                st.metric("Random Forest", f"{preds['rf']:.2f}%")
            
            # Cracking progression chart
            if 'progression' in st.session_state:
                prog = st.session_state['progression']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=prog['years'],
                    y=prog['cracking'],
                    mode='lines+markers',
                    name='Predicted Cracking',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                # Add threshold lines
                fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Good/Acceptable (15%)")
                fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Acceptable/Failure (30%)")
                
                fig.update_layout(
                    title="Cracking Progression Over Time",
                    xaxis_title="Pavement Age (Years)",
                    yaxis_title="Fatigue Cracking (%)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.markdown("## Sensitivity Analysis")
            st.caption("Shows how changing key parameters affects predicted cracking")
            
            base_inputs = st.session_state.get('base_inputs', {})
            
            # AC Thickness sensitivity
            ac_options = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
            ac_results = []
            for ac in ac_options:
                test_inputs = {
                    'Pavement_Age_Months': base_inputs['Pavement_Age_Months'],
                    'Cumulative_Monthly_ESALs': base_inputs['Cumulative_Monthly_ESALs'],
                    'AC_Thickness': ac,
                    'RAP_Percent': base_inputs['RAP_Percent'],
                    'A': base_inputs['A'],
                    'n': base_inputs['n'],
                    'AC_Modulus_ksi': base_inputs['AC_Modulus_ksi'],
                    'Base_Thickness': base_inputs['Base_Thickness'],
                    'Base_Modulus': base_inputs['Base_Modulus'],
                    'Subgrade_Modulus': base_inputs['Subgrade_Modulus']
                }
                t_xgb, t_lgb, t_rf = predict_cracking(test_inputs, scaler, xgb_model, lgb_model, rf_model)
                ac_results.append((ac, (t_xgb + t_lgb + t_rf) / 3))
            
            # Base Thickness sensitivity
            base_options = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0]
            base_results = []
            for base_thick in base_options:
                test_inputs = {
                    'Pavement_Age_Months': base_inputs['Pavement_Age_Months'],
                    'Cumulative_Monthly_ESALs': base_inputs['Cumulative_Monthly_ESALs'],
                    'AC_Thickness': base_inputs['AC_Thickness'],
                    'RAP_Percent': base_inputs['RAP_Percent'],
                    'A': base_inputs['A'],
                    'n': base_inputs['n'],
                    'AC_Modulus_ksi': base_inputs['AC_Modulus_ksi'],
                    'Base_Thickness': base_thick,
                    'Base_Modulus': base_inputs['Base_Modulus'],
                    'Subgrade_Modulus': base_inputs['Subgrade_Modulus']
                }
                t_xgb, t_lgb, t_rf = predict_cracking(test_inputs, scaler, xgb_model, lgb_model, rf_model)
                base_results.append((base_thick, (t_xgb + t_lgb + t_rf) / 3))
            
            # Mix Type sensitivity
            current_pg = st.session_state.get('selected_pg', selected_pg)
            mix_results = []
            for mix_type in ['Type B', 'Type C', 'Type D']:
                if (current_pg, mix_type) in binder_data:
                    test_A_raw, test_n = binder_data[(current_pg, mix_type)]
                    test_A_scaled = test_A_raw * 1e6
                    test_inputs = {
                        'Pavement_Age_Months': base_inputs['Pavement_Age_Months'],
                        'Cumulative_Monthly_ESALs': base_inputs['Cumulative_Monthly_ESALs'],
                        'AC_Thickness': base_inputs['AC_Thickness'],
                        'RAP_Percent': base_inputs['RAP_Percent'],
                        'A': test_A_scaled,
                        'n': test_n,
                        'AC_Modulus_ksi': base_inputs['AC_Modulus_ksi'],
                        'Base_Thickness': base_inputs['Base_Thickness'],
                        'Base_Modulus': base_inputs['Base_Modulus'],
                        'Subgrade_Modulus': base_inputs['Subgrade_Modulus']
                    }
                    t_xgb, t_lgb, t_rf = predict_cracking(test_inputs, scaler, xgb_model, lgb_model, rf_model)
                    mix_results.append((mix_type, (t_xgb + t_lgb + t_rf) / 3))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sensitivity Table")
                
                sens_data = []
                
                # AC Thickness
                for ac, crack in ac_results:
                    change = crack - avg_pred
                    sens_data.append({
                        'Parameter': 'AC Thickness',
                        'Value': f"{ac} in",
                        'Predicted Cracking': f"{crack:.2f}%",
                        'Change from Base': f"{change:+.2f}%"
                    })
                
                # Base Thickness
                for base_t, crack in base_results:
                    change = crack - avg_pred
                    sens_data.append({
                        'Parameter': 'Base Thickness',
                        'Value': f"{base_t} in",
                        'Predicted Cracking': f"{crack:.2f}%",
                        'Change from Base': f"{change:+.2f}%"
                    })
                
                # Mix Type
                for mix, crack in mix_results:
                    change = crack - avg_pred
                    sens_data.append({
                        'Parameter': 'Mix Type',
                        'Value': mix,
                        'Predicted Cracking': f"{crack:.2f}%",
                        'Change from Base': f"{change:+.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### What-If Summary")
                
                # Find best/worst for each parameter
                best_ac = min(ac_results, key=lambda x: x[1])
                worst_ac = max(ac_results, key=lambda x: x[1])
                
                best_base = min(base_results, key=lambda x: x[1])
                worst_base = max(base_results, key=lambda x: x[1])
                
                st.markdown(f"**AC Thickness:**")
                st.markdown(f"- Increasing from {worst_ac[0]} to {best_ac[0]} inches reduces cracking by {worst_ac[1] - best_ac[1]:.2f}%")
                
                st.markdown(f"**Base Thickness:**")
                st.markdown(f"- Increasing from {worst_base[0]} to {best_base[0]} inches reduces cracking by {worst_base[1] - best_base[1]:.2f}%")
                
                if mix_results:
                    best_mix = min(mix_results, key=lambda x: x[1])
                    worst_mix = max(mix_results, key=lambda x: x[1])
                    st.markdown(f"**Mix Type:**")
                    st.markdown(f"- Changing from {worst_mix[0]} to {best_mix[0]} reduces cracking by {worst_mix[1] - best_mix[1]:.2f}%")
    
    # ============================================================================
    # TAB 2: DESIGN COMPARISON TOOL
    # ============================================================================
    with tab2:
        st.title("Design Comparison Tool")
        st.markdown("Compare your base design with beefed-up and thinned alternatives")
        
        if 'base_inputs' not in st.session_state:
            st.info("Please run a prediction in the Design Input tab first to enable comparison.")
        else:
            base = st.session_state['base_inputs']
            total_esals = st.session_state.get('total_esals', 2000000)
            design_life = st.session_state['predictions']['design_life']
            
            st.markdown("### Design Alternatives")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Base Design**")
                st.write(f"AC Thickness: {base['AC_Thickness']} in")
                st.write(f"Base Thickness: {base['Base_Thickness']} in")
            
            with col2:
                st.markdown("**Beefed-Up Design**")
                beefed_ac = min(base['AC_Thickness'] + 1.0, 7.0)
                st.write(f"AC Thickness: {beefed_ac} in (+1.0 in)")
                st.write(f"Base Thickness: {base['Base_Thickness']} in")
            
            with col3:
                st.markdown("**Thinned Design**")
                thinned_ac = max(base['AC_Thickness'] - 1.0, 4.0)
                st.write(f"AC Thickness: {thinned_ac} in (-1.0 in)")
                st.write(f"Base Thickness: {base['Base_Thickness']} in")
            
            if st.button("Compare Designs", type="primary", key="compare_button"):
                # Generate progression for all three designs
                years = list(range(design_life + 1))
                
                progressions = {
                    'Base': [],
                    'Beefed-Up': [],
                    'Thinned': []
                }
                
                for year in years:
                    year_months = year * 12
                    year_esals = (total_esals / 240) * year_months if year > 0 else 0
                    
                    # Base design
                    base_inputs = {
                        'Pavement_Age_Months': year_months,
                        'Cumulative_Monthly_ESALs': year_esals,
                        'AC_Thickness': base['AC_Thickness'],
                        'RAP_Percent': base['RAP_Percent'],
                        'A': base['A'],
                        'n': base['n'],
                        'AC_Modulus_ksi': base['AC_Modulus_ksi'],
                        'Base_Thickness': base['Base_Thickness'],
                        'Base_Modulus': base['Base_Modulus'],
                        'Subgrade_Modulus': base['Subgrade_Modulus']
                    }
                    b_xgb, b_lgb, b_rf = predict_cracking(base_inputs, scaler, xgb_model, lgb_model, rf_model)
                    progressions['Base'].append((b_xgb + b_lgb + b_rf) / 3)
                    
                    # Beefed-up design
                    beefed_inputs = {
                        'Pavement_Age_Months': year_months,
                        'Cumulative_Monthly_ESALs': year_esals,
                        'AC_Thickness': beefed_ac,
                        'RAP_Percent': base['RAP_Percent'],
                        'A': base['A'],
                        'n': base['n'],
                        'AC_Modulus_ksi': base['AC_Modulus_ksi'],
                        'Base_Thickness': base['Base_Thickness'],
                        'Base_Modulus': base['Base_Modulus'],
                        'Subgrade_Modulus': base['Subgrade_Modulus']
                    }
                    bf_xgb, bf_lgb, bf_rf = predict_cracking(beefed_inputs, scaler, xgb_model, lgb_model, rf_model)
                    progressions['Beefed-Up'].append((bf_xgb + bf_lgb + bf_rf) / 3)
                    
                    # Thinned design
                    thinned_inputs = {
                        'Pavement_Age_Months': year_months,
                        'Cumulative_Monthly_ESALs': year_esals,
                        'AC_Thickness': thinned_ac,
                        'RAP_Percent': base['RAP_Percent'],
                        'A': base['A'],
                        'n': base['n'],
                        'AC_Modulus_ksi': base['AC_Modulus_ksi'],
                        'Base_Thickness': base['Base_Thickness'],
                        'Base_Modulus': base['Base_Modulus'],
                        'Subgrade_Modulus': base['Subgrade_Modulus']
                    }
                    t_xgb, t_lgb, t_rf = predict_cracking(thinned_inputs, scaler, xgb_model, lgb_model, rf_model)
                    progressions['Thinned'].append((t_xgb + t_lgb + t_rf) / 3)
                
                # Plot comparison
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=years, y=progressions['Base'],
                    mode='lines+markers',
                    name=f'Base ({base["AC_Thickness"]} in AC)',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=progressions['Beefed-Up'],
                    mode='lines+markers',
                    name=f'Beefed-Up ({beefed_ac} in AC)',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=years, y=progressions['Thinned'],
                    mode='lines+markers',
                    name=f'Thinned ({thinned_ac} in AC)',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                # Add threshold lines
                fig.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Good/Acceptable (15%)")
                fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Acceptable/Failure (30%)")
                
                fig.update_layout(
                    title="Design Comparison: Cracking Progression",
                    xaxis_title="Pavement Age (Years)",
                    yaxis_title="Fatigue Cracking (%)",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary table
                st.markdown("### Design Performance Summary")
                
                final_year_idx = design_life
                summary_data = {
                    'Design': ['Base', 'Beefed-Up', 'Thinned'],
                    'AC Thickness': [f"{base['AC_Thickness']} in", f"{beefed_ac} in", f"{thinned_ac} in"],
                    f'Cracking at {design_life} Years': [
                        f"{progressions['Base'][final_year_idx]:.2f}%",
                        f"{progressions['Beefed-Up'][final_year_idx]:.2f}%",
                        f"{progressions['Thinned'][final_year_idx]:.2f}%"
                    ]
                }
                
                # Add status
                statuses = []
                for design in ['Base', 'Beefed-Up', 'Thinned']:
                    crack = progressions[design][final_year_idx]
                    if crack < 15:
                        statuses.append('Good')
                    elif crack <= 30:
                        statuses.append('Acceptable')
                    else:
                        statuses.append('Early Failure')
                
                summary_data['Status'] = statuses
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("### Key Insights")
                
                base_final = progressions['Base'][final_year_idx]
                beefed_final = progressions['Beefed-Up'][final_year_idx]
                thinned_final = progressions['Thinned'][final_year_idx]
                
                reduction = base_final - beefed_final
                increase = thinned_final - base_final
                
                st.markdown(f"- Increasing AC thickness by 1 inch reduces cracking by **{reduction:.2f}%**")
                st.markdown(f"- Decreasing AC thickness by 1 inch increases cracking by **{increase:.2f}%**")
                st.markdown(f"- Difference between beefed-up and thinned designs: **{thinned_final - beefed_final:.2f}%**")
    
    # Footer
    st.markdown("---")
    st.markdown("*Pavement Fatigue Cracking Design Tool | Developed with Machine Learning*")
