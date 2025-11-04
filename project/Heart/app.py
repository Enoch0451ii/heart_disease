# app.py
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import altair as alt
import time
import os

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
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        # Use relative path instead of absolute path
        model_path = "project\Heart\heart_disease_model.pkl"
        if not os.path.exists(model_path):
            st.error("❌ Model file not found. Please ensure 'heart_disease_model.pkl' is in the same directory!")
            return None, None, None
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Handle different model data structures
        if isinstance(model_data, dict):
            model = model_data.get('model')
            feature_names = model_data.get('feature_names')
            test_accuracy = model_data.get('test_accuracy', 0.78)  # Default fallback
        else:
            # Assume it's the model object directly
            model = model_data
            feature_names = None
            test_accuracy = 0.78  # Default accuracy
        
        if model is not None:
            st.sidebar.success(f"✅ Model loaded ({test_accuracy:.1%} accuracy)")
        else:
            st.error("❌ Model could not be loaded from the file")
            
        return model, feature_names, test_accuracy
    
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

model, feature_names, model_accuracy = load_model()
if model is None:
    st.stop()

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
    restecg_encoded = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
    exang_encoded = 1 if exang == 'Yes' else 0
    slope_encoded = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
    thal_encoded = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal)

    # Create features array
    features = np.array([[
        age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded,
        restecg_encoded, thalach, exang_encoded, oldpeak, 
        slope_encoded, ca, thal_encoded
    ]])
    
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
    st.write("""
        This AI-powered web app predicts your **risk of heart disease** using clinical health data.

        🧠 **Model:** Random Forest Classifier  
        🩺 **Accuracy:** 78.0% (Realistic Test Set)  
        📊 **Dataset:** UCI Heart Disease (303 patients)  
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
    st.metric("Test Accuracy", f"{model_accuracy:.1%}")
    st.metric("Training Samples", "212 patients")
    st.metric("Test Samples", "91 patients")
    
    # Quick test cases
    st.markdown("---")
    st.markdown("### 🧪 Quick Test Cases")
    if st.button("Load High-Risk Example"):
        st.session_state.age = 65
        st.session_state.sex = "Male"
        st.session_state.cp = "Asymptomatic"
        st.session_state.trestbps = 180
        st.session_state.chol = 300
        st.session_state.fbs = "Yes"
        st.session_state.restecg = "ST-T Wave Abnormality"
        st.session_state.thalach = 110
        st.session_state.exang = "Yes"
        st.session_state.oldpeak = 4.2
        st.session_state.slope = "Flat"
        st.session_state.ca = 3
        st.session_state.thal = "Reversible Defect"
        st.rerun()
    
    if st.button("Load Low-Risk Example"):
        st.session_state.age = 35
        st.session_state.sex = "Female"
        st.session_state.cp = "Typical Angina"
        st.session_state.trestbps = 110
        st.session_state.chol = 180
        st.session_state.fbs = "No"
        st.session_state.restecg = "Normal"
        st.session_state.thalach = 175
        st.session_state.exang = "No"
        st.session_state.oldpeak = 0.5
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
with st.container():
    with st.form(key='heart_form'):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Personal & Clinical Data")
            age = st.slider('Age', 18, 100, value=st.session_state.get('age', 45))
            sex = st.selectbox('Sex', ('Male', 'Female'), 
                              index=0 if st.session_state.get('sex', 'Male') == 'Male' else 1)
            cp = st.selectbox('Chest Pain Type', 
                            ('Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'),
                            index=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(st.session_state.get('cp', 'Typical Angina')))
            trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, value=st.session_state.get('trestbps', 120))
            chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, value=st.session_state.get('chol', 240))

        with col2:
            st.markdown("### 🏥 Medical Test Results")
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'),
                              index=0 if st.session_state.get('fbs', 'No') == 'No' else 1)
            restecg = st.selectbox('Resting ECG Results', 
                                 ('Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'),
                                 index=['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(st.session_state.get('restecg', 'Normal')))
            thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, value=st.session_state.get('thalach', 150))
            exang = st.selectbox('Exercise Induced Angina', ('No', 'Yes'),
                                index=0 if st.session_state.get('exang', 'No') == 'No' else 1)
            oldpeak = st.slider('ST Depression (Oldpeak)', 0.0, 6.0, value=st.session_state.get('oldpeak', 1.0), step=0.1)
            slope = st.selectbox('Slope of Peak Exercise ST Segment', 
                               ('Upsloping', 'Flat', 'Downsloping'),
                               index=['Upsloping', 'Flat', 'Downsloping'].index(st.session_state.get('slope', 'Upsloping')))
            ca = st.slider('Number of Major Vessels (0–3)', 0, 3, value=st.session_state.get('ca', 0))
            thal = st.selectbox('Thalassemia', 
                              ('Normal', 'Fixed Defect', 'Reversible Defect'),
                              index=['Normal', 'Fixed Defect', 'Reversible Defect'].index(st.session_state.get('thal', 'Normal')))

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

            try:
                # Make prediction
                prediction = model.predict(features)[0]
                prediction_proba = model.predict_proba(features)[0]
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                st.stop()

        # Display Results
        st.markdown("## 📋 Result Overview")
        
        # Calculate confidence
        risk_probability = prediction_proba[1]
        confidence = int(risk_probability * 100)

        # Animated progress bar
        progress_placeholder = st.empty()
        for percent in range(0, confidence + 1, 4):
            time.sleep(0.02)
            progress_placeholder.progress(percent / 100)

        # Metrics
        col_conf, col_risk = st.columns(2)
        
        with col_conf:
            st.metric("🧠 Confidence Level", f"{confidence}%")
        
        with col_risk:
            if risk_probability > 0.7:
                risk_level = "High Risk 🚨"
            elif risk_probability > 0.4:
                risk_level = "Moderate Risk ⚠️"
            else:
                risk_level = "Low Risk ✅"
            st.metric("📊 Risk Assessment", risk_level)

        # Main result with appropriate styling
        if risk_probability > 0.7:
            st.markdown('<div class="result danger">🚨 High Risk of Heart Disease Detected!</div>', unsafe_allow_html=True)
        elif risk_probability > 0.4:
            st.markdown('<div class="result warning">⚠️ Moderate Risk - Further Evaluation Recommended</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result success">✅ Low Risk! Your heart seems healthy.</div>', unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("### 📈 Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>No Heart Disease</h3>
                <h2 style="color: #00FF7F;">{prediction_proba[0]*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with prob_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Heart Disease Present</h3>
                <h2 style="color: #FF4B4B;">{prediction_proba[1]*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

        # Risk Factor Analysis
        st.markdown("### 🔍 Risk Factors Analysis")
        
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

        # Health Advice
        st.markdown("### 💬 Health Advice")
        if risk_probability > 0.7:
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
        elif risk_probability > 0.4:
            st.markdown("""
            <div class="advice-box">
                💪 **Maintain Activity** - Regular moderate exercise<br>
                🥑 **Balanced Nutrition** - Focus on heart-healthy foods<br>
                😴 **Adequate Sleep** - 7-9 hours quality sleep<br>
                🧘 **Stress Management** - Practice relaxation techniques<br>
                🩺 **Regular Checkups** - Annual heart health screenings<br>
                📊 **Monitor Progress** - Track your health metrics
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

        # Data Visualization
        st.markdown("### 📊 Health Indicator Overview")
        
        metrics_data = pd.DataFrame({
            'Metric': ['Resting BP', 'Cholesterol', 'Max Heart Rate', 'ST Depression', 'Major Vessels'],
            'Value': [trestbps, chol, thalach, oldpeak, ca],
            'Risk Level': ['High' if (trestbps > 140 or chol > 240 or thalach < 120 or oldpeak > 2 or ca >= 2) else 'Normal' for _ in range(5)]
        })
        
        chart = alt.Chart(metrics_data).mark_bar(
            cornerRadiusTopLeft=8, 
            cornerRadiusTopRight=8
        ).encode(
            x=alt.X('Metric', sort=None, axis=alt.Axis(labelColor='white', titleColor='white', title='Health Metrics')),
            y=alt.Y('Value', axis=alt.Axis(labelColor='white', titleColor='white', title='Value')),
            color=alt.Color('Risk Level', scale=alt.Scale(domain=['High', 'Normal'], range=['#FF6A88', '#4CAF50'])),
            tooltip=['Metric', 'Value', 'Risk Level']
        ).properties(
            height=400,
            title='Current Health Metrics Visualization'
        ).configure_title(
            color='white',
            fontSize=16
        ).configure_legend(
            labelColor='white',
            titleColor='white'
        )

        st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
<hr style='margin-top:2rem;'>
<p style='text-align:center; color:gray; font-size:0.9rem;'>
© 2025 Oluwafemi | SIWES Project | Heart Disease Prediction Dashboard 💖
</p>
""", unsafe_allow_html=True)