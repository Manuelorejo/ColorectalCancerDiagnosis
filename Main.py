# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set page config
st.set_page_config(
    page_title="Gut Microbiome Analysis",
    page_icon="🦠",
    layout="wide"
)

# Title and introduction
st.title("Gut Microbiome Analysis")
st.markdown("""
This application analyzes gut microbiome profiles to predict study group classification.
Enter the microbiome abundance values below to get a prediction.
""")

# Define feature names
feature_names = [
    'd__Bacteria;p__Actinobacteriota;c__Coriobacteriia;o__Coriobacteriales;f__Coriobacteriaceae;g__Collinsella',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Bacteroides',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Phocaeicola',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Bacteroidaceae;g__Prevotella',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Marinifilaceae;g__Odoribacter',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Muribaculaceae;g__Limisoma',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Rikenellaceae;g__Alistipes',
    'd__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;f__Tannerellaceae;g__Parabacteroides',
    'd__Bacteria;p__Firmicutes;c__Bacilli;o__Lactobacillales;f__Streptococcaceae;g__Streptococcus',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Acetatifactor',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Agathobacter',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Anaerostipes',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Blautia_A',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Lachnospirales;f__Lachnospiraceae;g__Lachnospira',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Faecalibacterium',
    'd__Bacteria;p__Firmicutes_A;c__Clostridia;o__Oscillospirales;f__Ruminococcaceae;g__Ruminiclostridium_E',
    'd__Bacteria;p__Firmicutes_C;c__Negativicutes;o__Acidaminococcales;f__Acidaminococcaceae;g__Phascolarctobacterium',
    'd__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;o__Pseudomonadales;f__Pseudomonadaceae;g__Pseudomonas_E'
]

# Function to get simplified name for display
def simple_name(feature):
    parts = feature.split(';')
    if parts:
        genus = parts[-1].split('__')[-1]
        if genus == '':
            family = parts[-2].split('__')[-1]
            return f"{family} (unknown genus)"
        return genus
    return feature

# Load the model at the start of the app
try:
    model = joblib.load('LogisticRegression.sav')
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

# Create inputs for microbiome values
st.subheader("Enter Microbiome Values")

# Create three columns for layout
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

# Create sliders for each microbiome feature
sample_data = {}
for i, feature in enumerate(feature_names):
    col_idx = i % 3
    with columns[col_idx]:
        display_name = simple_name(feature)
        value = st.slider(
            display_name, 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1,
            step=0.01
        )
        sample_data[feature] = value

# Predict button
# Update the prediction results section
# Update the prediction results section
if st.button("Predict Group") and model_loaded:
    # Create DataFrame with input values
    input_df = pd.DataFrame([sample_data])
    
    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Map numerical prediction to group names
        group_map = {0: "Adenoma", 1: "Carcinoma", 2: "Control"}
        predicted_group = group_map[prediction]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Show prediction with explanation
        st.markdown(f"## Predicted Group: {predicted_group}")
        
        # Get confidence score (probability of the predicted class)
        confidence_score = probabilities[prediction] * 100
        
        # Display confidence score with a progress bar
        st.markdown("### Confidence Score")
        st.progress(confidence_score/100)
        st.markdown(f"**{confidence_score:.1f}%** confidence in this prediction")
        
        # Add explanation based on prediction
        st.markdown("### What this means:")
        if predicted_group == "Control":
            st.markdown("""
            Your microbiome profile is similar to individuals without colorectal abnormalities. 
            This suggests a lower risk for colorectal cancer based on the microbial markers analyzed.
            """)
        elif predicted_group == "Adenoma":
            st.markdown("""
            Your microbiome profile resembles that of patients with adenomas (precancerous polyps).
            Adenomas are benign growths that may develop into cancer over time if not removed.
            **Recommendation:** Discuss with your healthcare provider about appropriate screening procedures.
            """)
        elif predicted_group == "Carcinoma":
            st.markdown("""
            Your microbiome profile shows similarities to patients with colorectal carcinoma.
            This suggests potential presence of colorectal cancer based on microbial markers.
            **Recommendation:** Consult with a healthcare provider promptly for further clinical evaluation.
            """)
            
        # Display warning about the tool
        st.warning("Note: This is a screening tool only and not a diagnosis. Clinical evaluation by a healthcare professional is required for proper diagnosis.")
        
        # Create expandable section for detailed probability breakdown
        with st.expander("View Detailed Probability Breakdown"):
            for i, label in enumerate(model.classes_):
                group_name = group_map[label]
                probability = probabilities[i]
                
                col1, col2 = st.columns([7, 3])
                
                with col1:
                    st.markdown(f"**{group_name}**")
                    st.progress(probability)
                
                with col2:
                    st.markdown(f"**{probability*100:.1f}%**")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.info("Please make sure your model is compatible with this application.")
elif not model_loaded and st.button("Predict Group"):
    st.error("Cannot make prediction because model failed to load.")

# Add disclaimer
st.markdown("---")
st.caption("""
**Important Disclaimer**: This tool analyzes gut microbiome patterns only and should not replace medical advice, diagnosis, or treatment. 
The predictions are based on statistical patterns and require clinical validation. 
Please consult with a healthcare professional for proper screening and diagnosis of colorectal conditions.
""")