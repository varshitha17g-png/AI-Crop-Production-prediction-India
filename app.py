import streamlit as st
import joblib
import numpy as np

st.title("IN Crop Yield Prediction - India")
st.write("UCT ML Internship Project")

model = joblib.load('model.pkl')
le_crop = joblib.load('le_crop.pkl')
le_state = joblib.load('le_state.pkl')
scaler = joblib.load('scaler.pkl')



crop = st.selectbox("Select Crop", le_crop.classes_)
state = st.selectbox("Select State", le_state.classes_)
cost = st.number_input("Cost of Cultivation (₹/Hectare)", value=50000)

if st.button("Predict Yield"):
    try:
        # Transform chestunam - case sensitive
        crop_encoded = le_crop.transform([crop])[0]
        state_encoded = le_state.transform([state])[0]
        
        # IMPORTANT: 2D array cheyyali [[ ]]
        data = np.array([[crop_encoded, state_encoded, cost]])
        
        # Scale + Predict
        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)[0]
        
        st.success(f"Predicted Yield: {result:.2f} Quintal/Hectare")
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Check if crop/state exists in training data")
