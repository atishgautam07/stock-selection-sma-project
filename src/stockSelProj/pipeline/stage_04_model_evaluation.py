from stockSelProj.config.configuration import ConfigurationManager
from stockSelProj.components.model_evaluation import ModelEvaluation
from stockSelProj import logger
from pathlib import Path

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        if model_evaluation_config.trainModel:
            model_evaluation.run_register_model_xgb()
            model_evaluation.run_register_model_rf()
        else:
            pass



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e