import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dvclive import Live
import yaml

def load_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

# For test set creation
def prepare_data(data:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to prepare data: {e}")

def load_model(model_path: str) -> object:
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

def evaluate_model(model: object, X: pd.DataFrame, y: pd.Series) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml","r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["data_modelbuilding"]["n_estimators"]
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1_s = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        with Live(save_dvc_exp = True) as live:
            live.log("accuracy", acc)
            live.log("precision", prec)
            live.log("recall", recall)
            live.log("f1_score", f1_s)
            live.log("roc_auc", roc_auc)
            live.log("test_size", test_size)
            live.log("n_estimators", n_estimators)
        metrics_dict = {'accuracy': acc, 'precision': prec, 'recall': recall, 'f1_score': f1_s, 'roc_auc': roc_auc}
        return metrics_dict
    except Exception as e:
        raise Exception(f"Failed to evaluate model: {e}")

def save_metrics(metrics: dict, filepath: str) -> None:
    try:
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise Exception(f"Failed to save metrics: {e}")
    
def main():
    try:
        test_data_path = 'data/processed/test_data_processed.csv'
        model_path = 'models/model.pkl'
        metrics_path = 'reports/metrics.json'
        test_data = load_data(test_data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise Exception(f"Failed to execute main: {e}")
