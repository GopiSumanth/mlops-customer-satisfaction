import logging

import pandas as pd
from zenml import step
from model.model_development import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on ingested data

    Args:
        X_train (pd.DataFrame)
        y_train (pd.Series)

    Returns:
        RegressorMixin: Trained regression model
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported.")
    except Exception as e:
        logging.error("Error in training model", e)
        raise e