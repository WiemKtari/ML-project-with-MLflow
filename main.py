# main.py
import warnings
import mlflow

from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from mlProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from mlProject.pipeline.stage_03_model_trainer import ModelTrainingPipeline

# Ignore sklearn warnings for cleaner logs
warnings.filterwarnings("ignore")

# ===============================
# MLflow Configuration (DagsHub)
# ===============================

USERNAME = "wiem.ktari"
TOKEN = "..."# keep local, never commit
REPO = "ML-Project-with-MLflow"

TRACKING_URI = f"https://{USERNAME}:{TOKEN}@dagshub.com/{USERNAME}/{REPO}.mlflow"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("titanic-experiment")

# ===============================
# Main Pipeline Execution
# ===============================

if __name__ == "__main__":
    try:
        print(">>>> Stage: Data Ingestion >>>>")
        DataIngestionTrainingPipeline().main()
        print(">>>> Data Ingestion Completed!\n")

        print(">>>> Stage: Data Validation >>>>")
        DataValidationTrainingPipeline().main()
        print(">>>> Data Validation Completed!\n")

        print(">>>> Stage: Model Training >>>>")
        ModelTrainingPipeline().main()
        print(">>>> Model Training Completed!\n")

        print(">>>> All stages executed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise
