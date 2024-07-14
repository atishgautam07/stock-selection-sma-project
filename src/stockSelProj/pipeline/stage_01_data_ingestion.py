from stockSelProj.config.configuration import ConfigurationManager
from stockSelProj.components.data_ingestion import DataIngestion
from stockSelProj import logger

STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        if data_ingestion_config.fetchRepo:
            # Fetch All 3 datasets for all dates from APIs
            data_ingestion.fetch()
            # save data to a local dir
            data_ingestion.persist()
        else:
            # OR Load from disk
            data_ingestion.load()  


    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

