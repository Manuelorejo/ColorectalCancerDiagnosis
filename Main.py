# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(
    page_title="Colorectal Cancer Risk Prediction",
    page_icon="🔬",
    layout="wide"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    # Load the dataset
    df = pd.read_csv('Colorectal_Cancer.csv')
    
    # Convert gender to numeric for modeling
    df['gender_numeric'] = df['gender'].map({'M': 1, 'F': 0})
    
    # Prepare features and target
    X = df.drop(['patient_id', 'gender', 'colorectal_cancer'], axis=1)
    y = df['colorectal_cancer']
    
    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train Random Forest model with predefined best parameters
    best_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    best_model.fit(X_train_scaled, y_train)
    
    return best_model, scaler, list(X.columns)

# Load model and scaler
best_model, scaler, feature_names = load_model()

# Title and introduction
st.title("Colorectal Cancer Risk Prediction")
st.markdown("""
This application predicts the risk of colorectal cancer based on gut microbiome profiles.
Enter the patient information below to get a personalized risk assessment.
""")

# Display information about microbiome markers
with st.expander("About Gut Microbiome Markers"):
    st.markdown("""
    ### Key Microbiome Markers for Colorectal Cancer
    
    - **Fusobacterium nucleatum**: Enriched in CRC tissues, associated with disease progression
    - **Bacteroides fragilis**: Produces toxins that damage colon epithelium
    - **Escherichia coli (pks+)**: Produces genotoxins linked to CRC development
    - **Faecalibacterium prausnitzii**: Protective bacteria, lower levels associated with increased risk
    - **Other markers**: Include Peptostreptococcus anaerobius, Parvimonas micra, Gemella morbillorum
    
    The values range from 0 to 1, representing relative abundance in the microbiome.
    """)

# Create two columns for input
col1, col2 = st.columns(2)

# Patient demographic information
with col1:
    st.subheader("Patient Information")
    age = st.slider("Age", min_value=18, max_value=95, value=60)
    gender = st.radio("Gender", ["Male", "Female"])
    gender_numeric = 1 if gender == "Male" else 0

# Microbiome marker inputs
with col2:
    st.subheader("Microbiome Markers")
    
    # Create a default value dictionary based on average values
    # Ensure these features match and maintain the same order as feature_names
    microbiome_features = [f for f in feature_names if f not in ['age', 'gender_numeric']]
    
    default_values = {f: 0.3 for f in microbiome_features}
    # Set a higher default for the protective bacteria
    if 'Faecalibacterium_prausnitzii' in default_values:
        default_values['Faecalibacterium_prausnitzii'] = 0.6
    
    # Create sliders for each microbiome marker
    microbiome_values = {}
    for feature in microbiome_features:
        # Format the display name
        display_name = feature.replace('_', ' ')
        microbiome_values[feature] = st.slider(
            display_name, 
            min_value=0.0, 
            max_value=1.0, 
            value=default_values.get(feature, 0.3),
            step=0.01
        )

# Predict button
if st.button("Predict Risk"):
    # Create ordered dictionary matching training features
    patient_data = {}
    for feature in feature_names:
        if feature == 'age':
            patient_data[feature] = age
        elif feature == 'gender_numeric':
            patient_data[feature] = gender_numeric
        else:
            patient_data[feature] = microbiome_values.get(feature, 0.0)
    
    # Create DataFrame from patient data with correct feature order
    patient_df = pd.DataFrame([patient_data])
    
    # Ensure column order matches training data
    patient_df = patient_df[feature_names]
    
    # Scale features
    patient_scaled = scaler.transform(patient_df)
    
    # Predict probability
    risk_probability = best_model.predict_proba(patient_scaled)[0, 1]
    risk_percentage = risk_probability * 100
    
    # Determine risk category
    if risk_percentage < 30:
        risk_category = "Low"
        color = "green"
    elif risk_percentage < 70:
        risk_category = "Moderate"
        color = "orange"
    else:
        risk_category = "High"
        color = "red"
    
    # Display result
    st.markdown("---")
    st.subheader("Risk Assessment")
    
    # Create columns for results display
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        # Display risk meter
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Create gauge chart
        gauge_colors = ['green', 'yellow', 'orange', 'red']
        plt.pie(
            [1], 
            colors=['lightgray'],
            wedgeprops={'width': 0.4, 'edgecolor': 'white'},
            startangle=90,
            counterclock=False
        )
        plt.pie(
            [risk_probability, 1-risk_probability], 
            colors=[color, 'white'],
            wedgeprops={'width': 0.4, 'edgecolor': 'white'},
            startangle=90,
            counterclock=False
        )
        plt.title(f"{risk_percentage:.1f}% Risk", fontsize=16)
        plt.axis('equal')
        st.pyplot(fig)
        
        st.markdown(f"### {risk_category} Risk")
        
    with res_col2:
        st.markdown(f"""
        Based on the provided microbiome profile, this patient has a 
        **{risk_percentage:.1f}%** probability of colorectal cancer, 
        which is classified as **{risk_category} Risk**.
        """)
        
        # Recommendations based on risk
        st.subheader("Recommendations")
        if risk_category == "Low":
            st.markdown("""
            - Routine colorectal cancer screening as per age-appropriate guidelines
            - Maintain healthy lifestyle habits
            - No immediate additional testing required
            """)
        elif risk_category == "Moderate":
            st.markdown("""
            - Consider earlier screening colonoscopy
            - Evaluate other risk factors
            - Follow-up stool testing in 6-12 months
            """)
        else:
            st.markdown("""
            - Immediate referral for colonoscopy
            - Comprehensive cancer evaluation recommended
            - Close monitoring required
            """)
    
   

# Add information footer
st.markdown("---")
st.caption("""
**Disclaimer**: This tool is for educational purposes only and should not replace professional medical advice. 
Always consult with healthcare providers for diagnosis and treatment decisions.

Model: Random Forest classifier trained on gut microbiome data.
""")