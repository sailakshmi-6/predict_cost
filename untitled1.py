import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

# Set page title and layout
st.set_page_config(page_title="Repair Cost Prediction", layout="centered")

# Custom CSS Styling for a polished look
st.markdown("""
    <style>
    body {
        background-color: #f1f5f9;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        color: #003366;
        font-size: 40px;
        font-weight: 600;
        text-align: center;
        margin-top: 50px;
    }
    h3 {
        color: #0066cc;
        font-size: 24px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #009688;
        color: white;
        font-size: 18px;
        font-weight: 500;
        border-radius: 30px;
        padding: 15px 30px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00796b;
        box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.2);
    }
    .stSelectbox>div {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 10px;
    }
    .stSelectbox div span {
        color: #0066cc;  /* Set color for selected value */
        font-weight: bold;
    }
    .stAlert {
        background-color: #ffd54f;
        color: #e65100;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
    }
    .footer {
        text-align: center;
        color: #888888;
        font-size: 14px;
        padding: 10px;
        margin-top: 50px;
    }
    .prediction-result {
        font-size: 28px;
        color: #4caf50;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description for the app
st.markdown("<h1>Repair Cost Prediction for Manufacturing Products</h1>", unsafe_allow_html=True)
st.markdown("<h3>Predict repair costs based on defect type, severity, and inspection method.</h3>", unsafe_allow_html=True)

# Upload CSV file through the Streamlit interface
uploaded_file = st.file_uploader("Upload a CSV file containing your dataset", type="csv", label_visibility="collapsed")

# Check if the file is uploaded
if uploaded_file is not None:
    # Load CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Check for required columns in the dataset
    if 'repair_cost' not in df.columns:
        st.error("Dataset must contain a 'repair_cost' column for predictions.", icon="⚠️")
    elif 'defect_type' not in df.columns or 'severity' not in df.columns or 'inspection_method' not in df.columns:
        st.error("Dataset must contain 'defect_type', 'severity', and 'inspection_method' columns for features.", icon="⚠️")
    else:
        # Handle categorical features by encoding them
        le_type = LabelEncoder()
        le_severity = LabelEncoder()
        le_inspection = LabelEncoder()

        # Encoding categorical columns
        df['defect_type'] = le_type.fit_transform(df['defect_type'])
        df['severity'] = le_severity.fit_transform(df['severity'])
        df['inspection_method'] = le_inspection.fit_transform(df['inspection_method'])

        # Define features (X) and target (y)
        X = df[['defect_type', 'severity', 'inspection_method']]
        y = df['repair_cost']  # Target is now 'repair_cost'

        # Initialize and train the regression model
        model = RandomForestRegressor()
        model.fit(X, y)
        

        # Request user input for prediction
        st.subheader("Enter values for prediction:")

        # Custom selectboxes with enhanced visuals
        metric1 = st.selectbox('Defect Type', le_type.classes_, key="type")
        metric2 = st.selectbox('Severity', le_severity.classes_, key="severity")
        metric3 = st.selectbox('Inspection Method', le_inspection.classes_, key="inspection")

    

        # When the 'Predict' button is pressed, simulate action with a spinner
        if st.button('🔮 Predict Repair Cost', key="predict", use_container_width=True):
            with st.spinner("Predicting repair cost... please wait! ⏳"):
                time.sleep(2)  # Simulating delay to show spinner
                
                # Encode the user inputs using the same encoders as before
                encoded_input = np.array([le_type.transform([metric1])[0], 
                                          le_severity.transform([metric2])[0], 
                                          le_inspection.transform([metric3])[0]]).reshape(1, -1)
                
                # Make prediction
                predicted_cost = model.predict(encoded_input)

                # Display prediction result with dynamic styling
                st.markdown(f"<div class='prediction-result'>The predicted repair cost is: ${predicted_cost[0]:.2f}</div>", unsafe_allow_html=True)

                # Show a success message
                st.success("🎉 Prediction successful! The repair cost is ready!", icon="✅")

# Footer with custom message and style
st.markdown("<div class='footer'>Created with ❤️ by Your Name | Repair Cost Prediction App</div>", unsafe_allow_html=True)
