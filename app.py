import streamlit as st
from multiapp import MultiApp
from pages import csv_upload, user_input

app = MultiApp()

# Add all your application here
app.add_app("Homepage", lambda: st.write("Welcome to AquaPredict!"))
app.add_app("Dataset Upload", csv_upload.app)
app.add_app("Prediction", user_input.app)

# The main app
app.run()