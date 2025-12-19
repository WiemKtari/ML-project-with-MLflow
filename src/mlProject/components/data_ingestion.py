import os
import urllib.request as request
import zipfile
from pathlib import Path

from mlProject.entity.config_entity import DataIngestionConfig
from mlProject import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        os.makedirs(self.config.root_dir, exist_ok=True)

    def download_data(self):
        """Download file (ZIP or CSV) from source URL"""
        if not os.path.exists(self.config.local_data_file):
            try:
                filename, headers = request.urlretrieve(
                    url=self.config.source_url,
                    filename=self.config.local_data_file
                )
                logger.info(f"Downloaded data from {self.config.source_url}")
            except Exception as e:
                logger.error(f"Failed to download data: {e}")
                raise e
        else:
            logger.info(f"File already exists at {self.config.local_data_file}")

    def extract_zip_file(self):
        """Extract zip file into unzip directory, skip if None or not a zip"""
        unzip_path = self.config.unzip_dir

        # Skip extraction if no unzip_dir provided
        if not unzip_path:
            logger.info("No unzip_dir defined. Skipping extraction.")
            return

        # Only attempt extraction if file is a zip
        if not zipfile.is_zipfile(self.config.local_data_file):
            logger.info(f"File {self.config.local_data_file} is not a zip. Skipping extraction.")
            return

        os.makedirs(unzip_path, exist_ok=True)

        try:
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted zip file to {unzip_path}")
        except Exception as e:
            logger.error(f"Failed to extract zip file: {e}")
            raise e
