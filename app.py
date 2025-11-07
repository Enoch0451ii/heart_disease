# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt
import time
import os

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        # Debug: Check current directory and files
        current_dir = os.getcwd()
        
        
        # List files to see what's available
        files = os.listdir('.')
        
        
        # Check if file exists
        if not os.path.exists('heart_model.joblib'):
            st.error("❌ heart_model.joblib file not found!")
            st.info("Available files:")
            for file in files:
                st.info(f" - {file}")
            return None
        
        # Try to load the model
        model_data = joblib.load('heart_model.joblib')
        return model_data
        
    except FileNotFoundError:
        st.error("❌ Model file not found at 'heart_model.joblib'")
        st.info("Make sure the file is in your GitHub repository and deployed correctly.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("This might be a corrupted model file or missing dependencies.")
        return None
# Load model
model_data = load_model()
if model_data is None:
    st.stop()

model = model_data['model']
prob_index = model_data['probability_index']
accuracy = model_data['test_accuracy']

# Get the extreme test cases from training
extreme_high_risk = model_data.get('extreme_high_risk_case', [65, 1, 3, 180, 300, 1, 1, 110, 1, 4.2, 1, 3, 2])
extreme_low_risk = model_data.get('extreme_low_risk_case', [35, 0, 0, 110, 180, 0, 0, 175, 0, 0.5, 0, 0, 1])

# ----------------------------
# Prediction Function
# ----------------------------
def predict_heart_disease(input_features, model):
    """
    Prediction function with proper risk assessment
    """
    try:
        # Get probability predictions
        probability = model.predict_proba([input_features])[0]
        prediction = model.predict([input_features])[0]
        
        # Use the correct probability index for heart disease
        heart_disease_prob = probability[prob_index]
        
        # Risk thresholds - ADJUSTED for better separation
        if heart_disease_prob >= 0.65:  # 65% or higher (slightly lowered)
            risk_level = "🚨 HIGH RISK"
            risk_color = "red"
            recommendation = "Please consult a healthcare professional immediately."
        elif heart_disease_prob >= 0.35:  # 35% to 64%
            risk_level = "⚠️ MEDIUM RISK" 
            risk_color = "orange"
            recommendation = "Consider consulting a doctor for further evaluation."
        else:  # Below 35%
            risk_level = "✅ LOW RISK"
            risk_color = "green"
            recommendation = "Continue maintaining a healthy lifestyle!"
        
        return {
            'prediction': prediction,
            'probability': heart_disease_prob,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'healthy_probability': 1 - heart_disease_prob,
            'recommendation': recommendation,
            'raw_probabilities': probability
        }
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ----------------------------
# Initialize Session State
# ----------------------------
if 'age' not in st.session_state:
    st.session_state.age = 45
    st.session_state.sex = "Male"
    st.session_state.cp = "Typical Angina"
    st.session_state.trestbps = 120
    st.session_state.chol = 240
    st.session_state.fbs = "No"
    st.session_state.restecg = "Normal"
    st.session_state.thalach = 150
    st.session_state.exang = "No"
    st.session_state.oldpeak = 1.0
    st.session_state.slope = "Upsloping"
    st.session_state.ca = 0
    st.session_state.thal = "Normal"

# ----------------------------
# Input Validation Function
# ----------------------------
def validate_inputs(age, trestbps, chol, thalach, oldpeak, ca):
    """Validate medical input ranges"""
    warnings = []
    if age < 18 or age > 100:
        warnings.append("⚠️ Age should be between 18-100 years")
    if trestbps < 80 or trestbps > 200:
        warnings.append("⚠️ Resting BP should be between 80-200 mm Hg")
    if chol < 100 or chol > 600:
        warnings.append("⚠️ Cholesterol should be between 100-600 mg/dl")
    if thalach < 70 or thalach > 220:
        warnings.append("⚠️ Maximum heart rate should be between 70-220 bpm")
    if oldpeak < 0 or oldpeak > 6:
        warnings.append("⚠️ ST depression should be between 0-6")
    if ca < 0 or ca > 3:
        warnings.append("⚠️ Number of major vessels should be between 0-3")
    return warnings

# ----------------------------
# Data Preprocessing Function
# ----------------------------
def preprocess_inputs(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    """Convert all inputs to model-compatible format"""
    # Encode categorical variables
    sex_encoded = 1 if sex == 'Male' else 0
    cp_encoded = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
    fbs_encoded = 1 if fbs == 'Yes' else 0
    
    # Use consistent ECG option names
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg_encoded = restecg_options.index(restecg)
    
    exang_encoded = 1 if exang == 'Yes' else 0
    slope_encoded = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    thal_encoded = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)

    # Create features array
    features = [
        age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded,
        restecg_encoded, thalach, exang_encoded, oldpeak, 
        slope_encoded, ca, thal_encoded
    ]
    
    return features

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="❤️ Heart Disease Risk Predictor",
    page_icon="💓",
    layout="centered",
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
    <style>
        body {
            background: radial-gradient(circle at top left, #243B55, #141E30);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .main {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(18px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 0 25px rgba(255,255,255,0.1);
        }
        h1 {
            text-align: center;
            color: #FF6A88;
            font-size: 2.8rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background: linear-gradient(135deg, #FF4B4B, #FF6A88);
            color: white;
            border-radius: 12px;
            padding: 0.8rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton>button:hover {
            transform: scale(1.07);
            background: linear-gradient(135deg, #ff1f1f, #ff7b7b);
            box-shadow: 0 0 10px rgba(255,100,100,0.6);
        }
        .result {
            text-align: center;
            font-size: 1.3rem;
            padding: 1.2rem;
            border-radius: 10px;
            font-weight: bold;
            margin-top: 1rem;
        }
        .success {
            background-color: rgba(0, 255, 0, 0.15);
            color: #00FF7F;
            border: 2px solid #00FF7F;
        }
        .danger {
            background-color: rgba(255, 0, 0, 0.15);
            color: #FF4B4B;
            border: 2px solid #FF4B4B;
        }
        .warning {
            background-color: rgba(255, 193, 7, 0.15);
            color: #FFC107;
            border: 2px solid #FFC107;
        }
        .advice-box {
            background: rgba(255,255,255,0.1);
            padding: 1.5rem;
            border-radius: 15px;
            margin-top: 1rem;
            font-size: 1rem;
            line-height: 1.7;
            border-left: 4px solid #FF6A88;
        }
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem;
            text-align: center;
        }
        .risk-factor {
            background: rgba(255,100,100,0.2);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 4px solid #FF4B4B;
        }
        .good-factor {
            background: rgba(0, 255, 0, 0.1);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 4px solid #00FF7F;
        }
        .warning-factor {
            background: rgba(255, 193, 7, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin: 0.3rem 0;
            border-left: 4px solid #FFC107;
        }
        .validation-warning {
            background: rgba(255, 193, 7, 0.2);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #FFC107;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/7667/7667516.png", width=100)
    st.markdown("## 💡 About This Project")
    st.write(f"""
        This AI-powered web app predicts your **risk of heart disease** using clinical health data.

        🧠 **Model:** Random Forest Classifier  
        🩺 **Accuracy:** {accuracy:.1%}  
        📊 **Dataset:** UCI Heart Disease  
        ⚙️ **Tech Stack:** Streamlit + Scikit-learn  
        👨‍💻 **Developer:** Oluwafemi  
        🏫 **Project:** SIWES Final Defense Presentation
    """)
    st.markdown("---")
    st.info("""
    ⚠️ **Medical Disclaimer**  
    This tool is for educational purposes only.  
    Always consult healthcare professionals for medical diagnosis.
    """)
    
    # Model performance info
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("Test Accuracy", f"{accuracy:.1%}")
    st.metric("Training Samples", "≈250 patients")
    st.metric("Probability Index", f"{prob_index}")
    
    # Quick test cases - UPDATED WITH ALL THREE TEST CASES
    st.markdown("---")
    st.markdown("### 🧪 Quick Test Cases")
    
    if st.button("Load HIGH-RISK Example", key="high_risk_btn"):
        # Use the extreme high-risk values from training
        st.session_state.age = 70
        st.session_state.sex = "Male"
        st.session_state.cp = "Asymptomatic"
        st.session_state.trestbps = 200
        st.session_state.chol = 400
        st.session_state.fbs = "Yes"
        st.session_state.restecg = "Left Ventricular Hypertrophy"
        st.session_state.thalach = 100
        st.session_state.exang = "Yes"
        st.session_state.oldpeak = 5.0
        st.session_state.slope = "Downsloping"
        st.session_state.ca = 3
        st.session_state.thal = "Reversible Defect"
        st.rerun()
    
    if st.button("Load MODERATE-RISK Example", key="moderate_risk_btn"):
        # Use moderate-risk values
        st.session_state.age = 55
        st.session_state.sex = "Male"
        st.session_state.cp = "Atypical Angina"
        st.session_state.trestbps = 140
        st.session_state.chol = 240
        st.session_state.fbs = "No"
        st.session_state.restecg = "ST-T Abnormality"
        st.session_state.thalach = 140
        st.session_state.exang = "No"
        st.session_state.oldpeak = 1.5
        st.session_state.slope = "Flat"
        st.session_state.ca = 1
        st.session_state.thal = "Fixed Defect"
        st.rerun()
    
    if st.button("Load LOW-RISK Example", key="low_risk_btn"):
        # Use the extreme low-risk values from training
        st.session_state.age = 30
        st.session_state.sex = "Female"
        st.session_state.cp = "Typical Angina"
        st.session_state.trestbps = 100
        st.session_state.chol = 150
        st.session_state.fbs = "No"
        st.session_state.restecg = "Normal"
        st.session_state.thalach = 190
        st.session_state.exang = "No"
        st.session_state.oldpeak = 0.1
        st.session_state.slope = "Upsloping"
        st.session_state.ca = 0
        st.session_state.thal = "Normal"
        st.rerun()

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1>💖 Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.1rem; margin-bottom: 2rem;'>Fill in your medical details to check your heart health condition.</p>", unsafe_allow_html=True)

# ----------------------------
# Input Form
# ----------------------------
with st.form(key='heart_form'):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👤 Personal & Clinical Data")
        age = st.slider('Age', 18, 100, value=st.session_state.age)
        sex = st.selectbox('Sex', ('Male', 'Female'), 
                          index=0 if st.session_state.sex == 'Male' else 1)
        cp = st.selectbox('Chest Pain Type', 
                        ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'),
                        index=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(st.session_state.cp))
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, value=st.session_state.trestbps)
        chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, value=st.session_state.chol)

    with col2:
        st.markdown("### 🏥 Medical Test Results")
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'),
                          index=0 if st.session_state.fbs == 'No' else 1)
        
        # Use consistent ECG option names
        restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
        restecg = st.selectbox('Resting ECG Results', 
                             restecg_options,
                             index=restecg_options.index(st.session_state.restecg))
        
        thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, value=st.session_state.thalach)
        exang = st.selectbox('Exercise Induced Angina', ('No', 'Yes'),
                            index=0 if st.session_state.exang == 'No' else 1)
        oldpeak = st.slider('ST Depression (Oldpeak)', 0.0, 6.0, value=st.session_state.oldpeak, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                           ('Upsloping', 'Flat', 'Downsloping'),
                           index=['Upsloping', 'Flat', 'Downsloping'].index(st.session_state.slope))
        ca = st.slider('Number of Major Vessels (0–3)', 0, 3, value=st.session_state.ca)
        thal = st.selectbox('Thalassemia', 
                          ('Normal', 'Fixed Defect', 'Reversible Defect'),
                          index=['Normal', 'Fixed Defect', 'Reversible Defect'].index(st.session_state.thal))

    # Add proper submit button inside the form
    submitted = st.form_submit_button(label='🔍 Predict Heart Risk', use_container_width=True)

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:
    # Input validation
    validation_warnings = validate_inputs(age, trestbps, chol, thalach, oldpeak, ca)
    
    if validation_warnings:
        st.markdown("### ⚠️ Input Validation Warnings")
        for warning in validation_warnings:
            st.markdown(f'<div class="validation-warning">{warning}</div>', unsafe_allow_html=True)
        st.info("Please adjust your inputs and try again.")
    else:
        with st.spinner("🧠 Analyzing your heart health..."):
            time.sleep(1.5)

            # Preprocess inputs
            features = preprocess_inputs(
                age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal
            )

            # Make prediction
            result = predict_heart_disease(features, model)

        if result:
            # Display Results
            st.markdown("## 📋 Result Overview")
            
            # Calculate confidence
            risk_probability = result['probability']
            confidence = int(risk_probability * 100)

            # Animated progress bar
            progress_placeholder = st.empty()
            for percent in range(0, confidence + 1, 4):
                time.sleep(0.02)
                progress_placeholder.progress(percent / 100)

            # Metrics
            col_conf, col_risk = st.columns(2)
            
            with col_conf:
                st.metric("🧠 Disease Probability", f"{confidence}%")
            
            with col_risk:
                st.metric("📊 Risk Assessment", result['risk_level'])

            # Main result with appropriate styling
            if result['risk_level'] == "🚨 HIGH RISK":
                st.markdown(f'<div class="result danger">{result["risk_level"]} - Heart Disease Probability: {result["probability"]*100:.1f}%</div>', unsafe_allow_html=True)
            elif result['risk_level'] == "⚠️ MEDIUM RISK":
                st.markdown(f'<div class="result warning">{result["risk_level"]} - Heart Disease Probability: {result["probability"]*100:.1f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result success">{result["risk_level"]} - Heart Disease Probability: {result["probability"]*100:.1f}%</div>', unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("### 📈 Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Healthy Heart</h3>
                    <h2 style="color: #00FF7F;">{result['healthy_probability']*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with prob_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Heart Disease</h3>
                    <h2 style="color: #FF4B4B;">{result['probability']*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

            # Recommendation
            st.markdown("### 💬 Health Recommendation")
            st.info(f"**{result['recommendation']}**")

            # Debug information
            with st.expander("🔍 Technical Details"):
                st.write(f"Raw probabilities: {result['raw_probabilities']}")
                st.write(f"Using probability index: {prob_index}")
                st.write(f"Risk thresholds: HIGH ≥65%, MEDIUM 35-64%, LOW <35%")

            # Risk Factor Analysis - UPDATED FOR MODERATE RISK
            st.markdown("### 🔍 Risk Factors Analysis")
            
            if "MEDIUM" in result['risk_level']:
                # MODERATE RISK: Three-tier risk factor analysis
                risk_factors = []
                warning_factors = []
                good_factors = []

                # High Risk Factors (Red flags)
                if age > 60: 
                    risk_factors.append(f"Age over 60 ({age} years) - increased cardiovascular risk")
                if trestbps > 160: 
                    risk_factors.append(f"Stage 2 hypertension ({trestbps} mm Hg) - requires medication")
                if chol > 280: 
                    risk_factors.append(f"Very high cholesterol ({chol} mg/dL) - consider statin therapy")
                if thalach < 100: 
                    risk_factors.append(f"Very low exercise capacity ({thalach} bpm)")
                if oldpeak > 3: 
                    risk_factors.append(f"Significant ST depression ({oldpeak}) - possible ischemia")
                if exang == 'Yes': 
                    risk_factors.append("Exercise-induced angina - requires stress testing")
                if ca >= 2: 
                    risk_factors.append(f"Multiple coronary arteries affected ({ca})")
                if cp == 'Asymptomatic': 
                    risk_factors.append("Silent ischemia - most dangerous presentation")

                # Moderate Risk Factors (Yellow flags - your current profile)
                if 140 <= trestbps <= 159:
                    warning_factors.append(f"Stage 1 hypertension ({trestbps} mm Hg) - lifestyle modification needed")
                if 200 <= chol <= 279:
                    warning_factors.append(f"Borderline high cholesterol ({chol} mg/dL) - dietary changes recommended")
                if 120 <= thalach <= 139:
                    warning_factors.append(f"Reduced exercise tolerance ({thalach} bpm)")
                if 1.0 <= oldpeak <= 2.9:
                    warning_factors.append(f"Mild ST depression ({oldpeak}) - monitor during exercise")
                if ca == 1:
                    warning_factors.append(f"Single vessel involvement ({ca}) - early coronary disease")
                if cp == 'Atypical Angina':
                    warning_factors.append("Atypical chest pain - requires further evaluation")

                # Positive Factors
                if age < 50:
                    good_factors.append(f"Younger age ({age} years) - favorable prognosis")
                if trestbps < 120:
                    good_factors.append(f"Normal blood pressure ({trestbps} mm Hg)")
                if chol < 200:
                    good_factors.append(f"Desirable cholesterol level ({chol} mg/dL)")
                if thalach >= 160:
                    good_factors.append(f"Good exercise capacity ({thalach} bpm)")
                if oldpeak < 1.0:
                    good_factors.append(f"Minimal ST changes ({oldpeak})")
                if exang == 'No':
                    good_factors.append("No exercise-induced chest pain")
                if ca == 0:
                    good_factors.append("No major vessel disease")
                if cp == 'Typical Angina':
                    good_factors.append("Classic symptoms - easier to diagnose")

                # Display the analysis
                if risk_factors:
                    st.error("**🚨 High-Risk Factors (Require Immediate Attention):**")
                    for factor in risk_factors:
                        st.markdown(f'<div class="risk-factor">• {factor}</div>', unsafe_allow_html=True)

                if warning_factors:
                    st.warning("**⚠️ Moderate Risk Factors (Need Management):**")
                    for factor in warning_factors:
                        st.markdown(f'<div class="warning-factor">• {factor}</div>', unsafe_allow_html=True)

                if good_factors:
                    st.success("**✅ Positive Health Indicators:**")
                    for factor in good_factors:
                        st.markdown(f'<div class="good-factor">• {factor}</div>', unsafe_allow_html=True)

                # Specific recommendations based on moderate risk
                st.info("**🎯 Your Moderate Risk Management Focus:**")
                st.write("""
                • **Primary Goal**: Prevent progression to high risk category
                • **Secondary Goal**: Reduce modifiable risk factors by 25% in 6 months
                • **Key Metrics**: BP < 140/90, LDL < 100 mg/dL, BMI < 25
                • **Follow-up**: Repeat assessment in 6 months
                """)
                
            else:
                # Original risk factor analysis for HIGH and LOW risk
                risk_factors = []
                good_factors = []
                
                if age > 60: 
                    risk_factors.append(f"Age over 60 ({age} years)")
                else:
                    good_factors.append(f"Younger age ({age} years)")
                    
                if trestbps > 140: 
                    risk_factors.append(f"High blood pressure ({trestbps} mm Hg)")
                else:
                    good_factors.append(f"Normal blood pressure ({trestbps} mm Hg)")
                    
                if chol > 240: 
                    risk_factors.append(f"High cholesterol ({chol} mg/dl)")
                else:
                    good_factors.append(f"Normal cholesterol ({chol} mg/dl)")
                    
                if thalach < 120: 
                    risk_factors.append(f"Low maximum heart rate ({thalach})")
                else:
                    good_factors.append(f"Good maximum heart rate ({thalach})")
                    
                if oldpeak > 2: 
                    risk_factors.append(f"Significant ST depression ({oldpeak})")
                else:
                    good_factors.append(f"Normal ST depression ({oldpeak})")
                    
                if exang == 'Yes': 
                    risk_factors.append("Exercise induced angina")
                else:
                    good_factors.append("No exercise induced angina")
                    
                if ca >= 2: 
                    risk_factors.append(f"Multiple major vessels affected ({ca})")
                elif ca == 0:
                    good_factors.append("No major vessels affected")
                    
                if cp == 'Asymptomatic': 
                    risk_factors.append("Asymptomatic chest pain (most dangerous type)")
                elif cp == 'Typical Angina':
                    good_factors.append("Typical angina (less dangerous)")
                
                if risk_factors:
                    st.warning("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f'<div class="risk-factor">⚠️ {factor}</div>', unsafe_allow_html=True)
                
                if good_factors:
                    st.success("**Positive Health Indicators:**")
                    for factor in good_factors:
                        st.markdown(f'<div class="good-factor">✅ {factor}</div>', unsafe_allow_html=True)

            # Health Advice - UPDATED FOR MODERATE RISK
            st.markdown("### 💬 Health Advice")
            if result['risk_level'] == "🚨 HIGH RISK":
                st.markdown("""
                <div class="advice-box">
                    🏃 **Exercise Regularly** - Brisk walking 30 mins/day<br>
                    🥗 **Healthy Diet** - Low-fat, low-salt, high-fiber foods<br>
                    🚭 **Avoid Smoking** - Quit tobacco products<br>
                    🍷 **Limit Alcohol** - Moderate consumption only<br>
                    💊 **Monitor Health** - Regular BP and cholesterol checks<br>
                    🩺 **Consult Doctor** - Schedule a cardiology appointment immediately
                </div>
                """, unsafe_allow_html=True)
            elif result['risk_level'] == "⚠️ MEDIUM RISK":
                st.markdown("""
                <div class="advice-box">
                    **🩺 Moderate Risk Management Plan:**
                    
                    **Immediate Actions (Next 30 Days):**
                    • Schedule a comprehensive physical exam with your primary care physician
                    • Complete fasting lipid profile and HbA1c tests
                    • Consider a cardiac stress test for baseline assessment
                    • Begin blood pressure monitoring 2-3 times weekly
                    
                    **Lifestyle Modifications:**
                    • **Diet**: Adopt Mediterranean diet principles - focus on fish, olive oil, nuts, and vegetables
                    • **Exercise**: 150 minutes of moderate aerobic activity weekly (brisk walking, cycling)
                    • **Weight**: Aim for gradual weight loss if BMI > 25 (1-2 lbs per week)
                    • **Smoking**: Complete cessation if applicable
                    • **Alcohol**: Limit to 1 drink per day for men, 1 drink every other day for women
                    
                    **Medical Follow-up:**
                    • Re-evaluation in 3-6 months
                    • Consider low-dose aspirin therapy (consult your doctor)
                    • Discuss statin therapy if LDL > 130 mg/dL
                    
                    **Monitoring:**
                    • Track blood pressure weekly
                    • Annual cholesterol checks
                    • Regular weight and waist circumference monitoring
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="advice-box">
                    💪 **Maintain Activity** - Continue regular exercise<br>
                    🧘 **Manage Stress** - Practice relaxation techniques<br>
                    🥑 **Balanced Nutrition** - Maintain healthy eating habits<br>
                    😴 **Adequate Sleep** - 7-9 hours quality sleep nightly<br>
                    🩺 **Preventive Care** - Annual heart health screenings<br>
                    🚫 **Healthy Habits** - Avoid excessive junk food and stress
                </div>
                """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<hr style='margin-top:2rem;'>
<p style='text-align:center; color:gray; font-size:0.9rem;'>
© 2025 Oluwafemi | SIWES Project | Heart Disease Prediction Dashboard 💖
</p>
""", unsafe_allow_html=True)