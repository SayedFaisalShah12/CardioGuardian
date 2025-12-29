"""
Streamlit Web Application for CardioGuardian
Interactive web app for heart disease prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from predict import load_model_and_scaler, predict_single


# Page configuration
st.set_page_config(
    page_title="CardioGuardian - Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #f1faee;
        border: 2px solid #e63946;
        margin: 2rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ml_components():
    """Load model and scaler (cached for performance)"""
    try:
        model, scaler = load_model_and_scaler()
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">❤️ CardioGuardian</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #457b9d;">Heart Disease Prediction System</h2>', unsafe_allow_html=True)
    
    # Load model and scaler
    model, scaler, error = load_ml_components()
    
    if error:
        st.error(f"Error loading model: {error}")
        st.info("Please ensure you have trained the model first by running: python train_model.py")
        return
    
    # Sidebar for input features
    st.sidebar.header("📋 Patient Information")
    st.sidebar.markdown("Enter the patient's clinical parameters below:")
    
    # Feature inputs
    age = st.sidebar.slider("Age", min_value=18, max_value=100, value=50, step=1)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }[x]
    )
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.sidebar.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }[x]
    )
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }[x]
    )
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0, step=1)
    thal = st.sidebar.selectbox(
        "Thalassemia",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Unknown"
        }[x]
    )
    
    # Create feature dictionary
    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Patient Data Summary")
        
        # Display features in a nice format
        feature_display = pd.DataFrame({
            'Feature': [
                'Age', 'Sex', 'Chest Pain Type', 'Resting BP', 'Cholesterol',
                'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate',
                'Exercise Angina', 'ST Depression', 'ST Slope', 'Major Vessels', 'Thalassemia'
            ],
            'Value': [
                age, "Female" if sex == 0 else "Male",
                {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[cp],
                f"{trestbps} mm Hg", f"{chol} mg/dl",
                "Yes" if fbs == 1 else "No",
                {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[restecg],
                thalach,
                "Yes" if exang == 1 else "No",
                oldpeak,
                {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[slope],
                ca,
                {0: "Normal", 1: "Fixed Defect", 2: "Reversible", 3: "Unknown"}[thal]
            ]
        })
        st.dataframe(feature_display, use_container_width=True, hide_index=True)
    
    with col2:
        st.header("🔍 Prediction")
        
        # Predict button
        if st.button("🔮 Predict Heart Disease", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    # Make prediction
                    result = predict_single(model, scaler, features)
                    
                    # Display result
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    if result['prediction'] == 1:
                        st.error("⚠️ **Heart Disease Detected**")
                        st.warning("The model predicts that the patient may have heart disease. Please consult with a healthcare professional.")
                    else:
                        st.success("✅ **No Heart Disease Detected**")
                        st.info("The model predicts that the patient is unlikely to have heart disease.")
                    
                    # Display probabilities
                    if result['probability']:
                        st.markdown("---")
                        st.markdown("### Prediction Confidence")
                        col_prob1, col_prob2 = st.columns(2)
                        
                        with col_prob1:
                            st.metric(
                                "No Disease",
                                f"{result['probability']['no_disease']*100:.1f}%"
                            )
                        
                        with col_prob2:
                            st.metric(
                                "Heart Disease",
                                f"{result['probability']['disease']*100:.1f}%"
                            )
                        
                        # Progress bars
                        st.progress(result['probability']['no_disease'], text="No Disease Probability")
                        st.progress(result['probability']['disease'], text="Disease Probability")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About CardioGuardian")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        ### 📚 Model Information
        - **Algorithm**: Multiple models compared
        - **Best Model**: Selected based on accuracy
        - **Training Data**: Heart Disease Dataset
        """)
    
    with col_info2:
        st.markdown("""
        ### ⚠️ Important Disclaimer
        This tool is for educational purposes only.
        It should not replace professional medical advice.
        Always consult with healthcare professionals.
        """)
    
    with col_info3:
        st.markdown("""
        ### 🔬 Features Used
        - Age, Sex, Chest Pain Type
        - Blood Pressure, Cholesterol
        - ECG Results, Heart Rate
        - Exercise Parameters
        """)


if __name__ == "__main__":
    main()

