import os
import pytest
import json
import tensorflow as tf
from pathlib import Path
from deepClassifier.entity import EvaluationConfig
from deepClassifier.components import Evaluation
from deepClassifier.utils import get_size

class Test_Evaluation_evaluation:
    eval_config = EvaluationConfig (
        path_of_model="tests/data/training/model.h5",
        training_data="tests/data/data_ingestion/",
        params_image_size=[224, 224, 3],
        params_batch_size=1,
        score_json_path=''
    )

    def test_evaluation(self):
        evaluation = Evaluation(config=self.eval_config)
        evaluation.evaluation()
        assert os.path.exists(self.eval_config.path_of_model)
        assert os.path.isfile(Path(self.eval_config.path_of_model))
        assert get_size(Path(self.eval_config.path_of_model)) != 0

class Test_Evaluation_savescore:
    eval_config = EvaluationConfig (
        path_of_model="tests/data/training/model.h5",
        training_data="tests/data/data_ingestion/",
        params_image_size=[224, 224, 3],
        params_batch_size=1,
        score_json_path='tests/data/evaluate/scores.json'
    )

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
