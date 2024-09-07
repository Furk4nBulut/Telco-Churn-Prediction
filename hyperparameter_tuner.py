from sklearn.model_selection import GridSearchCV


class HyperparameterTuner:

    @staticmethod
    def tune_model(model, param_grid, X_train, y_train, cv=5, scoring='neg_mean_squared_error'):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Parameters:
        - model: The machine learning model to tune.
        - param_grid: Dictionary or list of dictionaries containing hyperparameters to search.
        - X_train: Training features.
        - y_train: Training labels.
        - cv: Number of cross-validation folds. Default is 5.
        - scoring: Scoring metric. Default is 'neg_mean_squared_error'.

        Returns:
        - best_model: The model with the best found hyperparameters.
        - best_params: The best hyperparameters found.
        - best_score: The best score achieved with the best hyperparameters.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best Hyperparameters: {best_params}")
        print(f"Best Cross-Validation Score: {best_score}")

        return best_model, best_params, best_score
