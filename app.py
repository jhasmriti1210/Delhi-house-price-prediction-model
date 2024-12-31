import streamlit as st
import pickle
import json
import numpy as np

# Load the model and columns
with open("delhi_home_prices_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("columns.json", "r") as columns_file:
    data_columns = json.load(columns_file)['data_columns']

# Extract location names
locations = [col for col in data_columns[3:]]  # Locations start from the 4th column onward

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #228B22;
        font-family: 'Arial', sans-serif;
    }
    .header {
        color: #FFA500;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #228B22;
        text-align: center;
        font-size: 24px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and header
st.markdown('<div class="header">Delhi Home Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get accurate property price predictions with just a few clicks!</div>',
            unsafe_allow_html=True)

# Add a banner image (Optional)
st.image("banner.png", use_container_width=True)


# Input fields
st.header("Enter Property Details:")
location = st.selectbox("Select Location", locations)
sqft = st.number_input("Enter Total Square Feet (e.g., 1500)", min_value=1, step=1)
bath = st.number_input("Enter Number of Bathrooms (e.g., 2)", min_value=1, step=1)
bhk = st.number_input("Enter Number of Bedrooms (e.g., 3)", min_value=1, step=1)

# Predict button
if st.button("Predict"):
    # Create feature array
    x = np.zeros(len(data_columns))
    x[0] = bhk
    x[1] = sqft
    x[2] = bath

    # Find the location index
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # Make prediction
    predicted_price = model.predict([x])[0]
    st.success(f"The estimated price of the property is: ₹{predicted_price:,.2f}")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #228B22; font-size: 25px;">Developed By Smriti Jha | © 2025 Delhi Property Price Predictor</div>',
    unsafe_allow_html=True)
