import pandas as pd
import joblib
import mlflow
from urllib.parse import urlparse
from pathlib import Path
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred)
        rec = recall_score(actual, pred)
        return acc, prec, rec

    def log_into_mlflow(self):
        # Charger les données et le modèle
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column]

        # Configurer MLflow
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Start MLflow run
        with mlflow.start_run():
            predictions = model.predict(test_x)
            acc, prec, rec = self.eval_metrics(test_y, predictions)

            # Sauvegarde locale
            scores = {"accuracy": acc, "precision": prec, "recall": rec}
            save_json(Path(self.config.metric_file_name), scores)

            # Log sur MLflow
            mlflow.log_metrics(scores)
            mlflow.log_params(self.config.all_params)

            # Log modèle
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="TitanicModel")
            else:
                mlflow.sklearn.log_model(model, "model")
