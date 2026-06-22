import streamlit as st
import joblib
import numpy as np

st.title("🇮🇳 Crop Yield Prediction - India")
st.write("UCT ML Internship Project")

model = joblib.load('model.pkl')
le_crop = joblib.load('le_crop.pkl')
le_state = joblib.load('le_state.pkl')
scaler = joblib.load('scaler.pkl')

crop = st.selectbox("Select Crop", le_crop.classes_)
state = st.selectbox("Select State", le_state.classes_)
cost = st.number_input("Cost of Cultivation (`/Hectare)", value=50000)

if st.button("Predict Yield"):
    data = np.array([[le_crop.transform([crop])[0], 
                     le_state.transform([state])[0], cost]])
    data = scaler.transform(data)
    result = model.predict(data)[0]
    st.success(f"Predicted Yield: {result:.2f} Quintal/Hectare")
