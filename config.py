import os

class Config:
    # Default configuration values
    DATASET_PATH = 'data/Telco-Customer-Churn.csv'
    OUTPUT_PATH = 'output/result.csv'
    OUTLIERS_LOWER_LIMIT = 0.35
    OUTLIERS_UPPER_LIMIT = 0.75
    TEST_SIZE = 0.40
    RANDOM_STATE = 46
    TARGET_COLUMN = 'Churn_Yes'
    METRICS = ['MSE', 'RMSE', 'MAE', 'R2 Score']

    # plot_metrics
    PLOT_METRICS = 'output/metrics_plot.png'
    PLOT_METRIC_COMPARISON = 'output/mse_comparison_plot.png'
    PLOT_MODEL_COMPARISON = 'output/model_comparison_plot.png'

    # Model-specific default hyperparameters
    HYPERPARAMETERS = {
        "LinearRegression": {},  # Linear Regression için hiperparametre gerekmez
        "RidgeRegression": {
            'alpha': [0.1, 1.0]  # Daha az değer
        },
        "LassoRegression": {
            'alpha': [0.1, 1.0]  # Daha az değer
        },
        "ElasticNet": {
            'alpha': [0.1, 1.0],
            'l1_ratio': [0.5]  # Sadece orta bir değer
        },
        "PolynomialRegression": {
            'polynomialfeatures__degree': [2, 3]  # Daha az derece
        },
        "DecisionTree": {
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        "RandomForestRegressor": {
            "max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]
        },
        "GradientBoostingRegressor": {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        },
        "SupportVectorMachine": {
            'C': [1.0],
            'epsilon': [0.1],
            'kernel': ['linear']  # 'rbf' kernel çıkarıldı
        },
        "KNearestNeighbors": {
            'n_neighbors': [3, 7],
            'weights': ['uniform'],
            'p': [2]  # Sadece Euclidean distance
        },
        "XGBoost": {
            "learning_rate": [0.1, 0.01, 0.001],
                  "max_depth": [5, 8, 12, 15, 20],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.5, 0.7, 1],
        },
        "LightGBM": {
            "learning_rate": [0.01, 0.1, 0.001],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]
        },
        "CatBoost": {
            "iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]
        }
    }

    @classmethod
    def set_config(cls, dataset_path=None, outliers_lower_limit=None, outliers_upper_limit=None, test_size=None, random_state=None, target_column=None, output_path=None, plot_metrics=None, plot_metric_comparison=None, plot_model_comparison=None):
        """Update configuration values."""
        if dataset_path is not None:
            cls.DATASET_PATH = dataset_path
        if outliers_lower_limit is not None:
            cls.OUTLIERS_LOWER_LIMIT = outliers_lower_limit
        if outliers_upper_limit is not None:
            cls.OUTLIERS_UPPER_LIMIT = outliers_upper_limit
        if test_size is not None:
            cls.TEST_SIZE = test_size
        if random_state is not None:
            cls.RANDOM_STATE = random_state
        if target_column is not None:
            cls.TARGET_COLUMN = target_column
        if output_path is not None:
            cls.OUTPUT_PATH = output_path
        if plot_metrics is not None:
            cls.PLOT_METRICS = plot_metrics
        if plot_metric_comparison is not None:
            cls.PLOT_METRIC_COMPARISON = plot_metric_comparison
        if plot_model_comparison is not None:
            cls.PLOT_MODEL_COMPARISON = plot_model_comparison
