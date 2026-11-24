
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide"
)

# Load saved files
model = joblib.load("final_aqi_model.pkl")
features = joblib.load("aqi_features.pkl")
defaults = joblib.load("feature_defaults.pkl")

st.title("🌍 Air Quality Index Prediction System")

st.write(
    "Machine Learning Based AQI Prediction"
)

st.divider()

st.subheader("Enter Air Pollution Values")

user_input = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(features):

    with (col1 if i % 2 == 0 else col2):

        user_input[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=float(defaults[feature]),
            key=feature
        )

st.divider()

if st.button("Predict AQI", type="primary"):

    input_df = pd.DataFrame(
        [user_input],
        columns=features
    )

    prediction = float(model.predict(input_df)[0])

    prediction = max(0, prediction)

    st.metric(
        label="Predicted AQI",
        value=f"{prediction:.2f}"
    )

    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 200:
        category = "Moderate"
    elif prediction <= 300:
        category = "Poor"
    elif prediction <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.subheader("Air Quality Category")
    st.success(category)

st.divider()

st.caption(
    "AQI Prediction System | Machine Learning Project"
)
