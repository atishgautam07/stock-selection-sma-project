from stockSelProj.config.configuration import ConfigurationManager
from stockSelProj.components.model_trainer import ModelTrainer
from stockSelProj import logger
from pathlib import Path

STAGE_NAME = "Model training stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train_rf()
        model_trainer_config.train_xgb()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e