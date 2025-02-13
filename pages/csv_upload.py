import streamlit as st
import pandas as pd
import mlflow
import io

# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/nipun5/AquaPredict_CDAC.mlflow")

# Load the latest model from MLflow
@st.cache_resource
def load_model():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("Best Model", stages=["Production"])
    run_id = versions[0].run_id
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/Best Model")

model = load_model()

def app():
    st.title("CSV Upload for Water Potability Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        # Store original columns for output
        original_data = input_data.copy()

        # Drop the target column 'Potability' or any extra columns if present
        if "Potability" in input_data.columns:
            input_data = input_data.drop(columns=["Potability"])

        # Define the expected columns
        expected_columns = [
            "ph", "Hardness", "Solids", "Chloramines",
            "Sulfate", "Conductivity", "Organic_carbon",
            "Trihalomethanes", "Turbidity"
        ]

        # Ensure only the required columns are used and drop others temporarily
        input_data = input_data[expected_columns]

        input_data = input_data.fillna(input_data.mean())

        # Perform predictions
        predictions = model.predict(input_data)

        # Add predictions to the original data
        original_data["Potable"] = ["Yes" if pred == 1 else "No" for pred in predictions]

        # Save the resulting DataFrame to a new CSV
        output_file_path = "output_with_predictions.csv"
        original_data.to_csv(output_file_path, index=False)

        # Calculate the overall probabilities
        potable_prob = sum(predictions) / len(predictions)
        non_potable_prob = 1 - potable_prob

        st.write(f"Potable Probability: {potable_prob}")
        st.write(f"Non-Potable Probability: {non_potable_prob}")

        st.download_button(
            label="Download Predictions",
            data=original_data.to_csv(index=False).encode('utf-8'),
            file_name='predictions.csv',
            mime='text/csv',
        )