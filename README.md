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

### Tkinter Desktop Application 🖥️
- **Tkinter App**:
   - A simple, user-friendly desktop app built with Tkinter.
   - Automatically fetches the latest model from the MLflow model registry.
   - Allows users to input data and receive potability predictions.

---

## 📦 Results and Analysis
- **Best Model**: Random Forest with mean imputation.
- **Optimal Hyperparameters**: `n_estimators=1000`, `max_depth=None`.

---

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
