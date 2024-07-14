import os
import pickle
from pathlib import Path
import mlflow
import numpy as np

class PredictionPipeline:
    def __init__(self, ml_uri, exp_name, model_name_rf, model_name_xgb):
        self.ml_uri = ml_uri
        self.exp_name = exp_name
        self.model_name_rf = model_name_rf
        self.model_name_xgb = model_name_xgb

        self.data_path = "artifacts/data_transformation"

    def load_pickle(self, filename):
        with open(filename, "rb") as f_in:
            return pickle.load(f_in)

    def load(self):
        """Load files from the local directory"""
        
        mlflow.set_tracking_uri(self.ml_uri)
        mlflow.set_experiment(self.exp_name)
        
        # client = MlflowClient(tracking_uri=self.ml_uri)
        model_version = "latest"

        model_uri_rf = f"models:/{self.model_name_rf}/{model_version}"
        model_uri_xgb = f"models:/{self.model_name_xgb}/{model_version}"
        
        self.best_rf_model = mlflow.sklearn.load_model(model_uri_rf)
        self.best_xgb_model = mlflow.xgboost.load_model(model_uri_xgb)
        

    def predict(self, pred_name:str):
        print('Making inference')

        self.X_all, self.y_all, self.X_train_valid, self.y_train_valid,self.X_train, self.y_train,self.X_test, self.y_test,self.X_valid, self.y_valid = self.load_pickle(os.path.join(self.data_path, "dfsForModel.pickle"))
        self.df_full = self.load_pickle(os.path.join(self.data_path, "dfOrigData.pickle"))

        y_pred_all_rf = self.best_rf_model.predict_proba(self.X_all)
        y_pred_all_class1_rf = [k[1] for k in y_pred_all_rf] #list of predictions for class "1"
        y_pred_all_class1_array_rf = np.array(y_pred_all_class1_rf) # (Numpy Array) np.array of predictions for class "1" , converted from a list

        y_pred_all_xgb = self.best_xgb_model.predict_proba(self.X_all)
        y_pred_all_class1_xgb = [k[1] for k in y_pred_all_xgb] #list of predictions for class "1"
        y_pred_all_class1_array_xgb = np.array(y_pred_all_class1_xgb) # (Numpy Array) np.array of predictions for class "1" , converted from a list

        self.df_full[pred_name + "_rf"] = y_pred_all_class1_array_rf
        self.df_full[pred_name + "_xgb"] = y_pred_all_class1_array_xgb

        self.df_full[pred_name] = (self.df_full[pred_name + "_rf"] + self.df_full[pred_name + "_xgb"]) / 2.0

        # define rank of the prediction
        self.df_full[f"{pred_name}_rank"] = self.df_full.groupby("Date")[pred_name].rank(method="first", ascending=False)


