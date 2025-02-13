import streamlit as st

# Set page configuration - This should be the very first Streamlit command
st.set_page_config(page_title="AquaPredict", layout="wide")

from multiapp import MultiApp
from pages import csv_upload, user_input, synthetic_data, homepage

app = MultiApp()

# Add all your application here
app.add_app("Homepage", homepage.app)
app.add_app("Dataset Upload", csv_upload.app)
app.add_app("Prediction", user_input.app)
app.add_app("Synthetic Data Generation", synthetic_data.app)

# The main app
app.run()