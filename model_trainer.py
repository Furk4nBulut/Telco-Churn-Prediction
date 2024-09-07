import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np
import config
from config import Config
from hyperparameter_tuner import HyperparameterTuner
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

class ModelTrainer:

    @staticmethod
    def preprocess_data(X):
        """Preprocess data including imputation, encoding, and scaling."""
        # Identify categorical and numerical columns
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(include=['number']).columns

        # Define preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        return preprocessor

    @staticmethod
    def train_and_evaluate_all_models(X_train, y_train, X_test, y_test):
        """Train and evaluate all models."""
        results = []
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "LassoRegression": Lasso(),
            "ElasticNet": ElasticNet(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SupportVectorMachine": SVR(),
            "KNearestNeighbors": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor()
        }

        for model_name, model in models.items():
            print(f"\nTraining and evaluating {model_name}...")

            # Convert to DataFrame for models with transformers
            if model_name == "PolynomialRegression":
                X_train_df = pd.DataFrame(X_train)
                X_test_df = pd.DataFrame(X_test)
                model.fit(X_train_df, y_train)
                metrics = ModelTrainer.evaluate_model(model, X_test_df, y_test, return_metrics=True)
            else:
                model.fit(X_train, y_train)
                metrics = ModelTrainer.evaluate_model(model, X_test, y_test, return_metrics=True)

            metrics['Model'] = model_name
            metrics['Type'] = 'Initial'
            results.append(metrics)

        return results
    @staticmethod
    def tune_and_evaluate_models(X_train, y_train, X_test, y_test):
        """Tune hyperparameters and evaluate all models."""
        results = []
        preprocessor = ModelTrainer.preprocess_data(X_train)
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "LassoRegression": Lasso(),
            "ElasticNet": ElasticNet(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SupportVectorMachine": Pipeline([
                ('preprocessor', preprocessor),
                ('model', SVR())
            ]),
            "KNearestNeighbors": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor()
        }

        for model_name, model in models.items():
            param_grid = config.Config.HYPERPARAMETERS.get(model_name.replace(' ', ''))
            if param_grid:
                print(f"\nTuning hyperparameters for {model_name}...")
                # Adjust parameter names for models within pipelines
                if 'Pipeline' in str(type(model)):
                    param_grid = {f'model__{key}': value for key, value in param_grid.items()}

                best_model, best_params, best_score = HyperparameterTuner.tune_model(
                    model,
                    param_grid, X_train, y_train
                )
                metrics = ModelTrainer.evaluate_model(best_model, X_test, y_test, return_metrics=True)
                metrics['Model'] = model_name
                metrics['Type'] = 'Tuned'
                results.append(metrics)
            else:
                print(f"No hyperparameters defined for {model_name}.")

        return results

    @staticmethod
    def evaluate_model(pipeline, X_test, y_test, return_metrics=False):
        """Evaluate the model and return performance metrics."""
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }

        if return_metrics:
            return metrics

        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")

    @staticmethod
    def export_results_to_csv(initial_results, tuned_results, filename=Config.OUTPUT_PATH):
        """Export results to a CSV file with initial and tuned results side by side."""
        # Initial results DataFrame
        initial_df = pd.DataFrame(initial_results)
        initial_df = initial_df.rename(columns=lambda x: f'Initial_{x}')

        # Tuned results DataFrame
        if tuned_results:
            tuned_df = pd.DataFrame(tuned_results)
            tuned_df = tuned_df.rename(columns=lambda x: f'Tuned_{x}')

            # Merge initial and tuned results
            combined_df = pd.concat([initial_df, tuned_df], axis=1)
        else:
            # If no tuned results, only export initial results
            combined_df = initial_df

        # Export to CSV
        combined_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

# Adjust your configuration to include hyperparameters if needed
