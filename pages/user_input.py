import streamlit as st
import pandas as pd
import mlflow

# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/nipun5/AquaPredict_CDAC.mlflow")

# Load the latest model from MLflow
@st.cache_resource
def load_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("Best Model", stages=["Production"])
    run_id = versions[0].run_id
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/Best Model")

model_rf = load_model()

def app():
    st.title('Water Potability Prediction')

    # User inputs with min and max values in labels
    ph = st.number_input('ph (0.0 to 14.0)', value=6.704635, min_value=0.0, max_value=14.0)
    Hardness = st.number_input('Hardness (0.0 to 500.0)', value=230.766940, min_value=0.0, max_value=500.0)
    Solids = st.number_input('Solids (0.0 to 50000.0)', value=9727.761716, min_value=0.0, max_value=50000.0)
    Chloramines = st.number_input('Chloramines (0.0 to 15.0)', value=5.943695, min_value=0.0, max_value=15.0)
    Sulfate = st.number_input('Sulfate (0.0 to 1000.0)', value=223.235816, min_value=0.0, max_value=1000.0)
    Conductivity = st.number_input('Conductivity (0.0 to 2000.0)', value=405.761571, min_value=0.0, max_value=2000.0)
    Organic_carbon = st.number_input('Organic Carbon (0.0 to 30.0)', value=12.826509, min_value=0.0, max_value=30.0)
    Trihalomethanes = st.number_input('Trihalomethanes (0.0 to 200.0)', value=74.385199, min_value=0.0, max_value=200.0)
    Turbidity = st.number_input('Turbidity (0.0 to 10.0)', value=3.422179, min_value=0.0, max_value=10.0)

    # Prediction
    if st.button('Predict'):
        # Preprocess user input
        data = {
            'ph': ph,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity
        }
        features = pd.DataFrame(data, index=[0])

        if model_rf is not None:
            # Make prediction
            prediction = model_rf.predict(features)

            # Output prediction
            st.subheader('Prediction')
            potability = 'Potable' if prediction[0] == 1 else 'Not Potable'
            st.write(potability)
        else:
            st.error("Model not loaded.")