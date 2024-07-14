from stockSelProj.config.configuration import ConfigurationManager
from stockSelProj.components.data_transformation import DataTransformation
from stockSelProj import logger
from pathlib import Path



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        if data_transformation_config.transfData:
            data_transformation.transform()
            data_transformation.persist()
            data_transformation.prepare_dataframe()
        else:
            data_transformation.load()
            data_transformation.prepare_dataframe()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e






