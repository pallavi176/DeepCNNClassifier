import os
import pytest
import json
from pathlib import Path
import tensorflow as tf
from deepClassifier.entity import (
    DataIngestionConfig, 
    PrepareBaseModelConfig,
    PrepareCallbacksConfig, 
    TrainingConfig,
    EvaluationConfig)
from deepClassifier.components import DataIngestion, PrepareBaseModel, PrepareCallback, Training, Evaluation
from deepClassifier.utils import get_size


class Test_Project:
    data_ingestion_config = DataIngestionConfig(
        root_dir="tests/data/data_ingestion/", 
        source_URL="https://raw.githubusercontent.com/pallavi176/raw_data/main/sample_data.zip", 
        local_data_file="tests/data/data_ingestion/data_integration.zip", 
        unzip_dir="tests/data/data_ingestion/")
    prepare_base_model_config = PrepareBaseModelConfig (
        root_dir='tests/data/prepare_base_model/',
        base_model_path='tests/data/prepare_base_model/base_model.h5',
        updated_base_model_path='tests/data/prepare_base_model/base_model_updated.h5',
        params_image_size=[224, 224, 3],
        params_learning_rate=0.01,
        params_include_top=False,
        params_weights='imagenet',
        params_classes=2)
    prepare_callbacks_config = PrepareCallbacksConfig (
        root_dir=Path('tests/data/prepare_callbacks/'),
        tensorboard_root_log_dir=Path('tests/data/prepare_callbacks/tensorboard_log_dir/'),
        checkpoint_model_filepath=Path('tests/data/prepare_callbacks/checkpoint_dir/model.h5'))
    training_config = TrainingConfig(
            root_dir=Path('tests/data/training/'),
            trained_model_path=Path('tests/data/training/model.h5'),
            updated_base_model_path=Path('tests/data/prepare_base_model/base_model_updated.h5'),
            training_data=Path("tests/data/data_ingestion/PetImages/"),
            params_epochs=1,
            params_batch_size=1,
            params_is_augmentation=True,
            params_image_size=[224, 224, 3])
    eval_config = EvaluationConfig (
        path_of_model="tests/data/training/model.h5",
        training_data="tests/data/data_ingestion/",
        params_image_size=[224, 224, 3],
        params_batch_size=1,
        score_json_path='tests/data/evaluate/scores.json'
    )

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

    def test_get_base_model(self):
        prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)
        model=prepare_base_model.get_base_model()
        assert isinstance(model, tf.keras.Model)
        assert os.path.exists(self.prepare_base_model_config.base_model_path)
        assert os.path.isfile(Path(self.prepare_base_model_config.base_model_path))
        assert get_size(Path(self.prepare_base_model_config.base_model_path)) != 0

    def test_update_base_model(self):
        prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)
        model=prepare_base_model.get_base_model()
        prepare_base_model.update_base_model(model=model)
        assert isinstance(model, tf.keras.Model)
        assert os.path.exists(self.prepare_base_model_config.updated_base_model_path)
        assert os.path.isfile(Path(self.prepare_base_model_config.updated_base_model_path))
        assert get_size(Path(self.prepare_base_model_config.updated_base_model_path)) != 0

    def test_get_tb_ckpt_callbacks(self):
        prepare_callbacks = PrepareCallback(config=self.prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()
        assert isinstance(callback_list, list)
        assert len(callback_list) == 2

    def test_get_base_model2(self):
        training = Training(config=self.training_config)
        training.get_base_model()
        assert os.path.exists(self.training_config.updated_base_model_path)

    def test_train(self):
        prepare_callbacks = PrepareCallback(config=self.prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training = Training(config=self.training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)
        assert os.path.exists(self.prepare_callbacks_config.tensorboard_root_log_dir)
        # assert os.path.exists(self.prepare_callbacks_config.checkpoint_model_filepath)
        assert os.path.exists(self.training_config.training_data)
        assert os.path.exists(self.training_config.trained_model_path)
        # assert os.path.isfile(self.prepare_callbacks_config.checkpoint_model_filepath)
        assert len(os.listdir(self.prepare_callbacks_config.tensorboard_root_log_dir)) > 0
        assert len(os.listdir(self.training_config.training_data)) > 0
        assert get_size(self.training_config.trained_model_path) != 0
        # assert get_size(self.prepare_callbacks_config.checkpoint_model_filepath) != 0

    def test_evaluation(self):
        evaluation = Evaluation(config=self.eval_config)
        evaluation.evaluation()
        assert os.path.exists(self.eval_config.path_of_model)
        assert os.path.isfile(Path(self.eval_config.path_of_model))
        assert get_size(Path(self.eval_config.path_of_model)) != 0

    def test_save_score(self):
        evaluation = Evaluation(config=self.eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        assert os.path.exists(Path(self.eval_config.score_json_path))
        assert os.path.isfile(Path(self.eval_config.score_json_path))
        assert get_size(Path(self.eval_config.score_json_path)) != 0
        with open(self.eval_config.score_json_path) as f:
            scores_data = json.load(f)
        assert scores_data
        assert isinstance(scores_data, dict)
        assert 'loss' in scores_data
        assert 'accuracy' in scores_data

