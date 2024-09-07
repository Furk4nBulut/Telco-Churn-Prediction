import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualization:

    @staticmethod
    def plot_metrics(results_df, metrics=['MSE', 'RMSE', 'MAE', 'R2 Score'], save_as=None):
        """Plot the specified metrics for initial and tuned models."""
        # Check if all specified metrics are present in the DataFrame
        missing_metrics = [metric for metric in metrics if metric not in results_df.columns]
        if missing_metrics:
            print(f"Warning: The following metrics are missing from the DataFrame: {missing_metrics}")
            return

        # Convert to long format for easier plotting
        df_long = pd.melt(results_df, id_vars=['Model', 'Type'], value_vars=metrics,
                          var_name='Metric', value_name='Value')

        plt.figure(figsize=(14, 8))
        sns.barplot(x='Model', y='Value', hue='Metric', data=df_long)
        plt.xticks(rotation=90)
        plt.title('Model Evaluation Metrics')
        plt.tight_layout()

        if save_as:
            plt.savefig(save_as)
            print(f"Plot saved as {save_as}")
        else:
            plt.show()

    @staticmethod
    def plot_metric_comparison(initial_df, tuned_df, metric, save_as=None):
        """Compare a specific metric before and after tuning."""
        if metric not in initial_df.columns or metric not in tuned_df.columns:
            print(f"Warning: Metric '{metric}' is not present in one of the DataFrames.")
            return

        df_initial = initial_df[['Model', metric]].copy()
        df_initial['Type'] = 'Initial'
        df_tuned = tuned_df[['Model', metric]].copy()
        df_tuned['Type'] = 'Tuned'

        df_combined = pd.concat([df_initial, df_tuned])

        plt.figure(figsize=(14, 8))
        sns.barplot(x='Model', y=metric, hue='Type', data=df_combined)
        plt.xticks(rotation=90)
        plt.title(f'{metric} Comparison Before and After Tuning')
        plt.tight_layout()

        if save_as:
            plt.savefig(save_as)
            print(f"Plot saved as {save_as}")
        else:
            plt.show()

    @staticmethod
    def plot_model_comparison(results_df, save_as=None):
        """Plot a comparison of all metrics for each model."""
        # Check if the required columns exist
        required_metrics = ['MSE', 'RMSE', 'MAE', 'R2 Score']
        for metric in required_metrics:
            if metric not in results_df.columns:
                print(f"Warning: Metric '{metric}' is missing from the DataFrame.")
                return

        plt.figure(figsize=(14, 8))
        for metric in required_metrics:
            sns.lineplot(x='Model', y=metric, hue='Type', data=results_df, marker='o')
        plt.xticks(rotation=90)
        plt.title('Comparison of Metrics for Each Model')
        plt.tight_layout()

        if save_as:
            plt.savefig(save_as)
            print(f"Plot saved as {save_as}")
        else:
            plt.show()