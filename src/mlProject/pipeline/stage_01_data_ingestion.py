from mlProject.components.data_ingestion import DataIngestion
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def main(self):
        try:
            logger.info(">>> Data Ingestion stage started <<<")
            
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            # Download the CSV file
            data_ingestion.download_data()
            
            # Attempt extraction (will skip if CSV)
            data_ingestion.extract_zip_file()

            logger.info(">>> Data Ingestion stage completed <<<")
        except Exception as e:
            logger.exception(e)
            raise e
