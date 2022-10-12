import os
import sys
import pandas as pd
from deepClassifier.config.configuration import ConfigurationManager
from deepClassifier.components.data_ingestion import DataIngestion


class Pipeline:
    def __init__(self, config: ConfigurationManager = ConfigurationManager()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise e

    def start_data_ingestion(self):
        try:
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise e

    def run_pipeline(self):
        # Data Ingestion
        self.start_data_ingestion()
