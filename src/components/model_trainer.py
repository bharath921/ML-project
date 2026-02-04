import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency
    CatBoostRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Support Vector Regressor": SVR(),
            }

            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "criterion": ["squared_error", "absolute_error"],
                    "max_features": ["sqrt", "log2"],
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7],
                    "weights": ["uniform", "distance"],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                    "subsample": [0.8, 1.0],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                },
                "Support Vector Regressor": {
                    "kernel": ["rbf", "poly"],
                    "C": [0.5, 1, 2],
                    "gamma": ["scale", "auto"],
                },
            }

            if XGBRegressor is not None:
                models["XGBRegressor"] = XGBRegressor()
                params["XGBRegressor"] = {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                }
            else:
                logging.warning("XGBoost not installed; skipping XGBRegressor model")

            if CatBoostRegressor is not None:
                models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False)
                params["CatBoosting Regressor"] = {
                    "depth": [4, 6, 8],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [100, 200, 500],
                }
            else:
                logging.warning("CatBoost not installed; skipping CatBoosting Regressor model")

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            if not model_report:
                raise CustomException("Model evaluation did not return any scores", sys)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Model training completed with R2 score: {r2_square}")
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
