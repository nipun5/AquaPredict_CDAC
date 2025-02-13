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
    st.markdown("Enter the water quality parameters below:")

    # User inputs with min and max values in labels
    ph_level = st.slider("pH Level 🌡️", 0.00, 14.00, 7.00)
    hardness = st.slider("Hardness (mg/L) 🧪", 0.00, 500.00, 250.00)
    chloramines = st.slider("Chloramines (mg/L) 🧴", 0.00, 15.00, 7.50)
    conductivity = st.slider("Conductivity (μS/cm) ⚡", 0.00, 800.00, 400.00)
    organic_carbon = st.slider("Organic Carbon (mg/L) 🌿", 0.00, 30.00, 15.00)
    trihalomethanes = st.slider("Trihalomethanes (μg/L) 🧫", 0.00, 120.00, 60.00)
    turbidity = st.slider("Turbidity (NTU) 🌊", 0.00, 10.00, 5.00)

    # Number inputs for solids and sulfate
    solids = st.number_input("Solids (ppm) 🧱", min_value=0.0, value=500.0)
    sulfate = st.number_input("Sulfate (mg/L) 🧂", min_value=0.0, value=250.0)
    
    # Prediction
    if st.button('Predict'):
        # Preprocess user input
        data = {
            'ph': ph_level,
            'Hardness': hardness,
            'Solids': solids,
            'Chloramines': chloramines,
            'Sulfate': sulfate,
            'Conductivity': conductivity,
            'Organic_carbon': organic_carbon,
            'Trihalomethanes': trihalomethanes,
            'Turbidity': turbidity
        }
        features = pd.DataFrame(data, index=[0])

        if model_rf is not None:
            # Make prediction
            prediction = model_rf.predict(features)

            # Output prediction
            st.subheader('Prediction')
            potability = 'Potable' if prediction[0] == 1 else 'Not Potable'
            
            if potability == 'Potable':
                st.success(potability)
                st.balloons()
            else:
                st.markdown(
                    f"""
                    <div style="background-color: red; padding: 10px; border-radius: 5px;">
                        <h3 style="color: white;">{potability}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Model not loaded.")
    