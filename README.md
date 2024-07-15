# Stock Selection and Prediction

## Overview
The application predicts the top 10 stocks every day with the highest probability of returning a profit in the next 5 days. Predictions are made by averaging the predicted probabilities of a tuned Random Forest and a tuned XGBoost model. The dependent variable is a binary indicator of whether a stock has positive growth over the next 5 days.

## Features
1. Data Sources: Daily indexes, commodities, forex data using yfinance, FRED, TA-Lib, and pandas_datareader.
2. Technical Indicators: Majority of momentum, volatility, and pattern indicators.
3. Machine Learning Models: Random Forest and XGBoost models.
4. Pipeline Stages: Data ingestion, transformation, model training, and evaluation.

## Project Structure

```bash
.
├── components
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   ├── model_hpo.py
│   ├── model_evaluation.py
│   ├── predict.py
├── config
│   ├── configuration.py
├── pipeline
│   ├── stage_01_data_ingestion.py
│   ├── stage_02_data_transformation.py
│   ├── stage_03_model_training.py
│   ├── stage_04_model_evaluation.py
├── config.yaml
├── app.py
├── main.py
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── README.md
```

## Configuration
### config.yaml
The config.yaml file contains all configurations, including:
1. Path to different artifact folders
2. MLflow URI
3. Boolean parameters to skip/run a particular stage in the pipeline
4. Parameters for hyperparameter tuning, such as the number of runs, number of top models to select, etc.

### ConfigurationManager Class
The ConfigurationManager class, located in config/configuration.py, reads configurations from the config.yaml file and provides methods to access these configurations.

## Pipeline Stages
## 1. Data Ingestion (stage_01_data_ingestion.py)
The data_ingestion.py script ingests, processes, and saves financial data. Here's a compact overview of its functionality:
### Methods
1. download_data(self):
- Downloads daily data for:
    - S&P 500 (^GSPC), Dow Jones (^DJI), Volatility Index (VIX) (^VIX)
    - NSE, BSE, NSE commodity index etc.
    - Gold, crude oil, brent oi, nautral gas, copper, aluminium. coal etc.
    - bitcoin, ethereum, xrp etc.
    - USD, GBP, EUR, JPY, AUD etc.
3. getIndiaStocks(self):
- filter list of India NSE ticker based on market cap, earnings growth, operating profit margin, cashflow, ROE, debt.
2. process_data(self):
- Merges downloaded data on date.
- Forward-fills missing values in specific fields.
persist(self):
- Saves data to local directory as Parquet files (tickers_df.parquet, indexes_df.parquet, macro_df.parquet).

## 2. Data Transformation (stage_02_data_transformation.py)
- Calls data_transformation.py.
- Adds technical indicators for the stocks.
- Combines all data points into one DataFrame merged by date.
#### Overview
- Initialization: Initializes with configuration parameters and reads ingested data.
- Feature Engineering: Adds various technical indicators, macroeconomic variables, and custom numerical features.
- Technical Indicators: Utilizes TA-Lib to compute a wide range of technical indicators.
- Data Processing: Merges different data sources, handles missing values, and prepares the final DataFrame.
- Saving Processed Data: Saves the transformed data for use in model training.
## 3. Model Training (stage_03_model_training.py)
- Calls model_hpo.py.
- Uses Hyperopt to tune the Random Forest and XGBoost models.
- Logs model parameters and metrics using MLflow.
- Summary of model_trainer.py
- The model_trainer.py script handles the training of machine learning models, including hyperparameter tuning.
### Overview
- Initialization: Reads configurations and initializes paths for data and model artifacts.
- Loading Data: Loads preprocessed data for training.
- Training Models: Trains both Random Forest and XGBoost models using the training data.
- Hyperparameter Tuning: Uses Hyperopt for hyperparameter optimization.
- Model Logging: Logs trained models and their metrics to MLflow for tracking.
## 4. Model Evaluation (stage_04_model_evaluation.py)
- Calls model_evaluation.py.
- Reads logged models for the best validation precision score.
- Selects the best model using the test set precision score and registers it with MLflow.
### Overview
- Initialization: Reads configuration settings and initializes paths for model artifacts and MLflow tracking.
- Loading Models: Loads models that were logged during the training phase using MLflow.
- Evaluation: Evaluates models based on precision scores using a test dataset.
- Model Selection: Selects the best-performing model based on evaluation metrics and registers it with MLflow.
- Saving Results: Logs evaluation metrics and details of the selected model to MLflow.

## Prediction (predict.py)
Calls the registered model for final predictions.

## Dependencies
yfinance
FRED
TA-Lib
pandas_datareader
pandas
numpy
scikit-learn
xgboost
hyperopt
mlflow
Flask

## Installation
To install the required packages and dependencies, run:

```bash
pipenv install --dev
```
This will create a virtual environment and install all dependencies as specified in the Pipfile and Pipfile.lock.

## Usage
### Running the Pipeline
You can run the entire pipeline using main.py or the train method in app.py.

1. Run using main.py
```bash
pipenv run python main.py
```
2. Run using Flask
Start the Flask application:
```bash
pipenv run python app.py
```
Then, you can trigger the pipeline by accessing:
```bash
http://0.0.0.0:9696/train
```
### Making Predictions
Predictions can be made using the predict_endpoint method in app.py, which calls predict.py.

1. Run using Flask
Start the Flask application:

```bash
pipenv run python app.py
```
Then, you can make predictions by accessing:

```bash
http://0.0.0.0:9696/predict
```

## MLflow
### Model Logging and Registry
- Model Training Stage: Runs hyperparameter tuning and logs all models using MLflow.
- Model Evaluation Stage: Uses MLflow to register the best models.
- Prediction: predict.py calls the registered model for final predictions.
### Running MLflow Locally
To launch the MLflow server locally, use the following command:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```
### Running MLflow Server on GCP
To run the MLflow server and log experiments on Google Cloud Platform (GCP), use the following command:

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://<username>:<pass>@<ip>:<host>/<db_name> --default-artifact-root <path_to_gs_bucket>
```
Replace <username>, <pass>, <ip>, <host>, <db_name>, and <path_to_gs_bucket> with your actual GCP and PostgreSQL details.

## Docker
### Building the Docker Image
To create a Docker image, use the following command:

```bash
docker build -t stock_selection-service:v1 .
```
### Running the Docker Container
To run the Docker container, use the following command:

```bash
docker run -it --rm -p 9696:9696 stock_selection-service:v1
```
This command will start the application and expose it on port 9696.

## License
This project is licensed under the MIT License - see the LICENSE file for details.