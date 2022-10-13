import os
import pytest
import tensorflow as tf
from pathlib import Path
from deepClassifier.entity import PrepareBaseModelConfig
from deepClassifier.components import PrepareBaseModel
from deepClassifier.utils import get_size

class Test_PrepareBaseModel_getbasemodel:
    prepare_base_model_config = PrepareBaseModelConfig (
        root_dir='tests/data/prepare_base_model/',
        base_model_path='tests/data/prepare_base_model/base_model.h5',
        updated_base_model_path='',
        params_image_size=[224, 224, 3],
        params_learning_rate='',
        params_include_top=False,
        params_weights='imagenet',
        params_classes=''
    )

    def test_get_base_model(self):
        prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)
        model=prepare_base_model.get_base_model()
        assert isinstance(model, tf.keras.Model)
        assert os.path.exists(self.prepare_base_model_config.base_model_path)
        assert os.path.isfile(Path(self.prepare_base_model_config.base_model_path))
        assert get_size(Path(self.prepare_base_model_config.base_model_path)) != 0

class Test_PrepareBaseModel_updatebasemodel:
    prepare_base_model_config = PrepareBaseModelConfig (
        root_dir='tests/data/prepare_base_model/',
        base_model_path='tests/data/prepare_base_model/base_model.h5',
        updated_base_model_path='tests/data/prepare_base_model/base_model_updated.h5',
        params_image_size=[224, 224, 3],
        params_learning_rate=0.01,
        params_include_top=False,
        params_weights='imagenet',
        params_classes=2
    )

    def test_update_base_model(self):
        prepare_base_model = PrepareBaseModel(config=self.prepare_base_model_config)
        model=prepare_base_model.get_base_model()
        prepare_base_model.update_base_model(model=model)
        assert isinstance(model, tf.keras.Model)
        assert os.path.exists(self.prepare_base_model_config.updated_base_model_path)
        assert os.path.isfile(Path(self.prepare_base_model_config.updated_base_model_path))
        assert get_size(Path(self.prepare_base_model_config.updated_base_model_path)) != 0
