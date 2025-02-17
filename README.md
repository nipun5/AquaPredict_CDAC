# 🌊 Water Potability Prediction with MLOps

Welcome! This project demonstrates an end-to-end MLOps workflow to predict water potability using machine learning. We use tools like **MLflow** for tracking, **DVC** for versioning, and **Tkinter** for creating a desktop application.

## 📈 Project Overview

- **Objective**: Predict water potability based on water quality metrics.
- **Goal**: Build an MLOps pipeline that tracks experiments, versions data and models, and deploys a desktop app for easy predictions.

---

## 🔄 Project Workflow

1. **Experiment Setup**: Use a pre-configured Cookiecutter template and initialize Git for version control.
2. **MLflow Tracking**: Log experiments and model metrics on DagsHub using MLflow.
3. **DVC Pipeline**: Set up data versioning with DVC and build a robust ML pipeline.
4. **Model Registration**: Register the best model in MLflow’s registry for easy deployment.
5. **Desktop Application**: Create a Tkinter app that fetches the latest model from MLflow and performs predictions.

---

## 🛋️ Running the Application

Launch the Streamlit application by running:

```
streamlit run Implementation/app.py
```

This command opens the interactive dashboard in your default web browser. Use the sidebar menu to navigate between different pages such as homepage, synthetic dataset, prediction, and CSV upload.

---

## 🛂 Results and Analysis

- **Best Model**: Random Forest with mean imputation.
- **Optimal Hyperparameters**: `n_estimators=1000`, `max_depth=None`.

---
