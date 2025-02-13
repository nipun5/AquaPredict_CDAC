import streamlit as st

def app():
    st.title("🌊 Water Potability Prediction with MLOps")
    st.write("""
    Welcome! This project demonstrates an end-to-end MLOps workflow to predict water potability using machine learning. We use tools like MLflow for tracking, DVC for versioning, and Tkinter for creating a desktop application.
    """)

    st.header("📈 Project Overview")
    st.write("""
    **Objective:** Predict water potability based on water quality metrics.
    **Goal:** Build an MLOps pipeline that tracks experiments, versions data and models, and deploys a desktop app for easy predictions.
    """)

    st.header("🔄 Project Workflow")
    st.write("""
    - **Experiment Setup:** Use a pre-configured Cookiecutter template and initialize Git for version control.
    - **MLflow Tracking:** Log experiments and model metrics on DagsHub using MLflow.
    - **DVC Pipeline:** Set up data versioning with DVC and build a robust ML pipeline.
    - **Model Registration:** Register the best model in MLflow’s registry for easy deployment.
    - **Desktop Application:** Create a Tkinter app that fetches the latest model from MLflow and performs predictions.
    """)

    # st.image("https://github.com/your-repo-path/water-potability-image.png", caption="Water Potability")

if __name__ == "__main__":
    app()