import os
import pytest
from pathlib import Path
from deepClassifier.entity import DataIngestionConfig
from deepClassifier.components import DataIngestion


class Test_DataIngestion:
    data_ingestion_config = DataIngestionConfig(
        root_dir="tests/data/data_ingestion/", 
        source_URL="https://raw.githubusercontent.com/pallavi176/raw_data/main/sample_data.zip", 
        local_data_file="tests/data/data_ingestion/data_integration.zip", 
        unzip_dir="tests/data/data_ingestion/")

    def test_download(self):
        data_ingestion = DataIngestion(config=self.data_ingestion_config)
        data_ingestion.download_file()
        assert os.path.exists(self.data_ingestion_config.local_data_file)

    def test_unzip(self):
        data_ingestion = DataIngestion(config=self.data_ingestion_config)
        data_ingestion.unzip_and_clean()
        assert os.path.isdir(Path("tests/data/data_ingestion/PetImages"))
        assert os.path.isdir(Path("tests/data/data_ingestion/PetImages/Cat"))
        assert os.path.isdir(Path("tests/data/data_ingestion/PetImages/Dog"))



        
