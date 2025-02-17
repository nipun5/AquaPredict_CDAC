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

## 📂 Project Structure
This project follows a structured workflow to streamline the MLOps process:

### Setup
- Install project structure with Cookiecutter.
- Initialize **Git** and push to **GitHub**.

### Experiment Tracking
1. **DagsHub + MLflow**:
   - Log experiments on DagsHub.
   - Track model metrics, parameters, and artifacts.
   
2. **Experiment Execution**:
   - **Experiment 1**: Baseline model with Random Forest.
   - **Experiment 2**: Multiple models (e.g., Logistic Regression, XGBoost).
   - **Experiment 3**: Test mean vs. median imputation for missing values.
   - **Experiment 4**: Hyperparameter tuning on Random Forest.

### DVC Pipeline
1. **Data Versioning**:
   - Set up DVC for versioning data on a local disk (or cloud if preferred).
   
2. **Pipeline Stages**:
   - **Data Collection**: Gather and structure data.
   - **Data Preprocessing**: Handle missing values (mean imputation).
   - **Model Building**: Train a Random Forest model.
   - **Model Evaluation**: Track performance metrics with MLflow.

### Model Registration
- **MLflow Registry**:
  - Register the best model with optimal parameters and metadata.
  - Deploy the model using **FastAPI** or **Streamlit** for predictions.

## Running the Application

### Streamlit Application
Launch the Streamlit application by running:

```bash
streamlit run Implementation/app.py 

## 📦 Results and Analysis
- **Best Model**: Random Forest with mean imputation.
- **Optimal Hyperparameters**: `n_estimators=1000`, `max_depth=None`.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
