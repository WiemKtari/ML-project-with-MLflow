from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    def __init__(self, root_dir, source_url, local_data_file, unzip_dir=None):
        self.root_dir = root_dir
        self.source_url = source_url
        self.local_data_file = local_data_file
        self.unzip_dir = unzip_dir


class DataValidationConfig:
    def __init__(self, root_dir, STATUS_FILE, unzip_data_dir, all_schema):
        self.root_dir = root_dir
        self.STATUS_FILE = STATUS_FILE
        self.unzip_data_dir = unzip_data_dir
        self.all_schema = all_schema


class DataTransformationConfig:
    def __init__(self, root_dir, ingested_data_dir, transformed_train_dir, transformed_test_dir):
        self.root_dir = root_dir
        self.ingested_data_dir = ingested_data_dir
        self.transformed_train_dir = transformed_train_dir
        self.transformed_test_dir = transformed_test_dir


class ModelTrainerConfig:
    def __init__(self, root_dir, model_file):
        self.root_dir = root_dir
        self.model_file = model_file


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str



