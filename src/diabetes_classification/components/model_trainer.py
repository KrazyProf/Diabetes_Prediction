import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from diabetes_classification.exception.exception import CustomException
from diabetes_classification.logging.logger import logging

from diabetes_classification.utils.utlis import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def get_best_params(self, model, param_grid, X_train, y_train):
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)
            return grid_search.best_params_
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'max_depth': [3, 5, 7]
                },
                "Logistic Regression": {
                    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
                    'C': [0.1, 1.0, 10.0],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                "XGBClassifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'max_depth': [3, 5, 7]
                },
                "CatBoost Classifier": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.05],
                    'iterations': [100, 200, 300]
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05]
                }
            }

            best_params = {}
            for model_name in models.keys():
                logging.info(f"Finding best parameters for {model_name}")
                best_params[model_name] = self.get_best_params(models[model_name], params[model_name], X_train, y_train)

            logging.info(f"Best parameters found: {best_params}")

            tuned_models = {
                "Random Forest": RandomForestClassifier(**best_params["Random Forest"]),
                "Decision Tree": DecisionTreeClassifier(**best_params["Decision Tree"]),
                "Gradient Boosting": GradientBoostingClassifier(**best_params["Gradient Boosting"]),
                "Logistic Regression": LogisticRegression(**best_params["Logistic Regression"]),
                "XGBClassifier": XGBClassifier(**best_params["XGBClassifier"]),
                "CatBoost Classifier": CatBoostClassifier(**best_params["CatBoost Classifier"]),
                "AdaBoost Classifier": AdaBoostClassifier(**best_params["AdaBoost Classifier"]),
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=tuned_models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = tuned_models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predicted)
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
