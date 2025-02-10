import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/nipun5/AquaPredict_CDAC.mlflow")

dagshub.init(repo_owner='nipun5', repo_name='AquaPredict_CDAC', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)