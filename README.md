# deep Classifier project
DeepCNNClassifier

## workflow

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config.
6. Update the components
7. Update the pipeline
8. Test run pipeline stage
9. run tox for testing your package
10. Update the dvc.yaml
11. run "dvc repro" for running all the stages in pipeline

![img](https://raw.githubusercontent.com/pallavi176/DeepCNNClassifier/main/docs/images/Data%20Ingestion%402x%20(1).png)

#mlflow tutorials:
https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html

mlflow server command -

mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./artifacts \
--host 0.0.0.0 -p 1234

STEP 1: Set the env variable | Get it from dagshub -> remote tab -> mlflow tab

MLFLOW_TRACKING_URI=https://dagshub.com/pallavi176/DeepCNNClassifier.mlflow \
MLFLOW_TRACKING_USERNAME=pallavi176 \
MLFLOW_TRACKING_PASSWORD=<> \

STEP 2: install mlflow

STEP 3: Set remote URI

STEP 4: Use context manager of mlflow to start run and then log metrics, params and model


## Sample data for testing-
https://raw.githubusercontent.com/pallavi176/raw_data/main/sample_data.zip

Steps for testing:

Step1: Create below folders:
tests/data/data_ingestion
tests/data/prepare_base_model
tests/data/prepare_callbacks
tests/data/training
tests/data/evaluate

Step 2: Download sample data from path: https://raw.githubusercontent.com/pallavi176/raw_data/main/sample_data.zip
& copy to folder: tests/data/data_ingestion

Step 3: Execute the below command to run test cases:

pytest -v

