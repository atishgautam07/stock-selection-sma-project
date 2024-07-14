from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    fetchRepo: bool
    resPath: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transfData: bool


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    num_trials: int
    cv: int
    data_path: Path
    mlflow_uri: str
    hpo_exp_rf: str
    hpo_exp_xgb: str
    trainModel: bool
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    top_n: int
    ml_uri: str
    hpo_exp_rf: str
    hpo_exp_xgb: str
    exp_name: str
    trainModel: bool