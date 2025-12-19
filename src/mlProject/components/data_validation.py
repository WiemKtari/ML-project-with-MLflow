import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig
from mlProject import logger
from pathlib import Path

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        """Check if all required columns exist in the dataset"""
        data_file = Path(self.config.unzip_data_dir)
        df = pd.read_csv(data_file)
        missing_columns = [
            col for col in self.config.all_schema.keys() if col not in df.columns
        ]

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            status = False
        else:
            logger.info("All required columns are present.")
            status = True

        # Save status to file
        Path(self.config.STATUS_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(str(status))

        return status
