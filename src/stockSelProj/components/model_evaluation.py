import os
import mlflow
import pickle
import numpy as np
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  precision_score
from stockSelProj import logger
from stockSelProj.entity.config_entity import (ModelEvaluationConfig)



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def load_pickle(self, filename):
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)


    def run_register_model_rf(self):

        mlflow.set_tracking_uri(self.config.ml_uri)
        mlflow.set_experiment(self.config.exp_name)
        mlflow.sklearn.autolog()
        client = MlflowClient(tracking_uri=self.config.ml_uri)

        self.X_all, self.y_all, self.X_train_valid, self.y_train_valid,self.X_train, self.y_train,self.X_test, self.y_test,self.X_valid, self.y_valid = self.load_pickle(os.path.join(self.config.data_path, "dfsForModel.pickle"))
        
        # Retrieve the top_n model runs and log the models
        logger.info("Retrieve the top_n model runs and log the models.")
        experiment = client.get_experiment_by_name(self.config.hpo_exp_rf)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.precision DESC"]
        )
        logger.info(len(runs))
        logger.info("logging top_n models wiht test metrics.")
        
        for run in runs:
            logger.info((str(run.info.run_id), str(run.data.metrics), str(run.data.params)))
            modelPath = client.download_artifacts(run_id=run.info.run_id, path="model")
            pipeLine = self.load_pickle(os.path.join(modelPath, "model.pkl"))
            

            with mlflow.start_run():

                mlflow.set_tag("model", "rf_topN_models")
                mlflow.log_params(run.data.params)
                
                pipeLine.fit(self.X_train_valid.to_numpy(), self.y_train_valid.to_numpy())
                logger.info("Evaluate model on the validation and test sets")
                val_score = precision_score(self.y_valid.to_numpy(), pipeLine.predict(self.X_valid.to_numpy()))
                mlflow.log_metric("val_score", val_score)
                test_score = precision_score(self.y_test.to_numpy(), pipeLine.predict(self.X_test.to_numpy()))
                mlflow.log_metric("test_score", test_score)
                mlflow.sklearn.log_model(pipeLine, artifact_path="model")

        logger.info("Selecting the model with the lowest test score")
        experiment = client.get_experiment_by_name(self.config.exp_name)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            filter_string='tags.model="rf_topN_models"',
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.test_score DESC"]
        )[0]

        # Register the best model
        logger.info("Registering the best RF model")
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="best-model-rf")


    def run_register_model_xgb(self):

        mlflow.set_tracking_uri(self.config.ml_uri)
        mlflow.set_experiment(self.config.exp_name)
        mlflow.xgboost.autolog()
        client = MlflowClient(tracking_uri=self.config.ml_uri)

        self.X_all, self.y_all, self.X_train_valid, self.y_train_valid,self.X_train, self.y_train,self.X_test, self.y_test,self.X_valid, self.y_valid = self.load_pickle(os.path.join(self.config.data_path, "dfsForModel.pickle"))
        
        # Retrieve the top_n model runs and log the models
        logger.info("Retrieve the top_n model runs and log the models.")
        experiment = client.get_experiment_by_name(self.config.hpo_exp_xgb)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.precision DESC"]
        )
        logger.info(len(runs))
        logger.info("logging top_n models wiht test metrics.")
        
        for run in runs:
            logger.info((str(run.info.run_id), str(run.data.metrics), str(run.data.params)))
            modelPath = client.download_artifacts(run_id=run.info.run_id, path="model")
            pipeLine = mlflow.xgboost.load_model(modelPath)#os.path.join(modelPath, "model.xgb"))

            with mlflow.start_run():

                mlflow.set_tag("model", "xgb_topN_models")
                mlflow.log_params(run.data.params)
                
                logger.info("Evaluate model on the validation and test sets")
                val_score = precision_score(self.y_valid.to_numpy(), pipeLine.predict(self.X_valid))
                mlflow.log_metric("val_score", val_score)
                test_score = precision_score(self.y_test.to_numpy(), pipeLine.predict(self.X_test))
                mlflow.log_metric("test_score", test_score)
                mlflow.xgboost.log_model(pipeLine, artifact_path="model")

        logger.info("Selecting the model with the lowest test score")
        experiment = client.get_experiment_by_name(self.config.exp_name)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            filter_string='tags.model="xgb_topN_models"',
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=self.config.top_n,
            order_by=["metrics.test_score DESC"]
        )[0]

        # Register the best model
        logger.info("Registering the best XGB model")
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="best-model-xgb")
