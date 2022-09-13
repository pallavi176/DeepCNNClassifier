import os
from deepClassifier.pipeline.pipeline import Pipeline
from deepClassifier.config.configuration import ConfigurationManager


def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()
        print("main function execution completed.")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()