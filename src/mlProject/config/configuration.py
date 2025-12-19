import os
from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories
from mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])
        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        # Path to ingested data (from data ingestion)
        ingested_data_dir = os.path.join(self.config.artifacts_root, "data_ingestion")

        return DataTransformationConfig(
            root_dir=config.root_dir,
            ingested_data_dir=ingested_data_dir,
            transformed_train_dir=config.transformed_train_dir,
            transformed_test_dir=config.transformed_test_dir
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=config.root_dir,
            model_file=config.model_file
        )


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.xxx
        schema = self. schema. TARGET_COLUMN
        create_directories([config.root_dir])
        model_evaluation_config = ModelEvaluationConfig(
        root_dir=config.root_dir,
        test_data_path=config.test_data_path,
        model_path = config.model_path,
        all_params=params,
        metric_file_name = config.metric_file_name,
        target_column = schema.name,
        mlflow_uri="https://dagshub.com/wiem.ktari/ML-Project-with-MLflow.mlflow",

        )
        return model_evaluation_config

