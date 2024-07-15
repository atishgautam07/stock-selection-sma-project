from stockSelProj.constants import *
from stockSelProj.utils.common import read_yaml, create_directories
from stockSelProj.entity.config_entity import (DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            fetchRepo=config.FETCH_REPO,
            resPath=config.resPath,
            calcTickers=config.calcTickers
        )

        return data_ingestion_config    
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            transfData=config.TRANSFORM_DATA
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            num_trials=config.num_trials,
            cv = config.cv,
            data_path = config.data_path,
            mlflow_uri = config.mlflow_uri,
            hpo_exp_rf = config.hpo_exp_rf,
            hpo_exp_xgb = config.hpo_exp_xgb,
            trainModel = config.trainModel
        )

        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            top_n=config.top_n,
            ml_uri=config.ml_uri,
            hpo_exp_rf=config.hpo_exp_rf,
            hpo_exp_xgb=config.hpo_exp_xgb,
            exp_name=config.exp_name,
            trainModel = config.trainModel
        )

        return model_evaluation_config



    