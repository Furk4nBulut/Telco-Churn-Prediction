import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import config


class DataPreprocessor:
    @staticmethod
    def check_missing_values(data):
        """Check for missing values in the dataset."""
        return data.isnull().sum()

    @staticmethod
    def impute_missing_values(data):
        """Impute missing values using mean for numerical and most frequent for categorical features."""
        numerical_features = data.select_dtypes(include=['number']).columns
        categorical_features = data.select_dtypes(include=['object']).columns

        # Impute numerical features with mean
        imputer_num = SimpleImputer(strategy='mean')
        data[numerical_features] = imputer_num.fit_transform(data[numerical_features])

        # Impute categorical features with most frequent value
        imputer_cat = SimpleImputer(strategy='most_frequent')
        data[categorical_features] = imputer_cat.fit_transform(data[categorical_features])

        return data

    @staticmethod
    def drop_missing_values(data):
        """Drop rows with missing values."""
        return data.dropna()

    @staticmethod
    def outlier_thresholds(dataframe, variable):
        """Determine the outlier thresholds for a given variable."""
        quartile1 = dataframe[variable].quantile(config.Config.OUTLIERS_LOWER_LIMIT)
        quartile3 = dataframe[variable].quantile(config.Config.OUTLIERS_UPPER_LIMIT)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    @staticmethod
    def check_outlier(dataframe, col_name):
        """Check if there are outliers in the given column."""
        low_limit, up_limit = DataPreprocessor.outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

    @staticmethod
    def replace_with_thresholds(dataframe, col_name):
        """Replace outliers with the thresholds."""
        low_limit, up_limit = DataPreprocessor.outlier_thresholds(dataframe, col_name)
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

    @staticmethod
    def preprocess_data(data):
        """Perform data preprocessing including outlier handling, feature engineering, and scaling."""

        # Convert TotalCharges to numeric, forcing errors to NaN, then fill with 0
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce').fillna(0)

        # Impute missing values
        data = DataPreprocessor.impute_missing_values(data)

        # Separate categorical and numerical columns
        categorical_data = data.select_dtypes(include=['object'])
        numerical_data = data.select_dtypes(exclude=['object'])

        # Check and handle outliers
        for col in numerical_data.columns:
            if DataPreprocessor.check_outlier(data, col):
                DataPreprocessor.replace_with_thresholds(data, col)

        # Feature Engineering
        data.loc[(data["tenure"] >= 0) & (data["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
        data.loc[(data["tenure"] > 12) & (data["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
        data.loc[(data["tenure"] > 24) & (data["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
        data.loc[(data["tenure"] > 36) & (data["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
        data.loc[(data["tenure"] > 48) & (data["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
        data.loc[(data["tenure"] > 60) & (data["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

        # Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
        data["NEW_Engaged"] = data["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

        # Herhangi bir destek, yedek veya koruma almayan kişiler
        data["NEW_noProt"] = data.apply(
            lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (
                    x["TechSupport"] != "Yes") else 0, axis=1)

        # Aylık sözleşmesi bulunan ve genç olan müşteriler
        data["NEW_Young_Not_Engaged"] = data.apply(
            lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

        # Kişinin toplam aldığı servis sayısı
        data['NEW_TotalServices'] = (data[['PhoneService', 'InternetService', 'OnlineSecurity',
                                           'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                           'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)

        # Herhangi bir streaming hizmeti alan kişiler
        data["NEW_FLAG_ANY_STREAMING"] = data.apply(
            lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

        # Kişi otomatik ödeme yapıyor mu?
        data["NEW_FLAG_AutoPayment"] = data["PaymentMethod"].apply(
            lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

        # Ortalama aylık ödeme
        data["NEW_AVG_Charges"] = data["TotalCharges"] / (data["tenure"] + 1)

        # Güncel Fiyatın ortalama fiyata göre artışı
        data["NEW_Increase"] = data["NEW_AVG_Charges"] / data["MonthlyCharges"]

        # Servis başına ücret
        data["NEW_AVG_Service_Fee"] = data["MonthlyCharges"] / (data['NEW_TotalServices'] + 1)

        # One-Hot Encoding
        data = pd.get_dummies(data, columns=categorical_data.columns, drop_first=True)

        return data

    @staticmethod
    def convert_columns_to_numeric(df):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def encode_categorical_features(df):
        categorical_columns = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df
