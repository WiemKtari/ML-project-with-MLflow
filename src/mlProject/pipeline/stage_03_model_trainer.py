# src/mlProject/pipeline/stage_03_model_trainer.py
from mlProject.logger import logger
from mlProject.components.model_trainer import ModelTrainer
from mlProject.config.configuration import ConfigurationManager
import os
import mlflow

class ModelTrainingPipeline:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.train_path = os.path.join(self.artifacts_dir, "data_transformation", "train", "train.csv")
        self.test_path = os.path.join(self.artifacts_dir, "data_transformation", "test", "test.csv")
        self.model_save_path = os.path.join(self.artifacts_dir, "model_trainer", "model.joblib")

    def main(self):
        STAGE_NAME = "Model Training stage"
        try:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

            # Load hyperparameters from params.yaml
            config_manager = ConfigurationManager()
            # Assuming your params.yaml has a section like:
            # RandomForest:
            #   n_estimators: 100
            #   max_depth: 10
            #   min_samples_split: 2
            #   min_samples_leaf: 1
            #   random_state: 42
            model_params = config_manager.params.RandomForest

            # Explicitly log parameters in MLflow
            mlflow.log_params(model_params)

            # Initialize trainer with hyperparameters and save path
            trainer = ModelTrainer(
                train_path=self.train_path,
                test_path=self.test_path,
                model_params=model_params,
                save_model_path=self.model_save_path
            )

            # Train model and save
            model = trainer.train_model()

            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        except Exception as e:
            logger.exception(e)
            raise e


# Example usage
if __name__ == "__main__":
    pipeline = ModelTrainingPipeline()
    pipeline.main()
