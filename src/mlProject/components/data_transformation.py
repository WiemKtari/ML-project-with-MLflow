import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from mlProject import logger
from sklearn.pipeline import Pipeline


class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.input_path = os.path.join(self.config.ingested_data_dir, "titanic.csv")
        self.transformed_train_dir = self.config.transformed_train_dir
        self.transformed_test_dir = self.config.transformed_test_dir
        os.makedirs(self.transformed_train_dir, exist_ok=True)
        os.makedirs(self.transformed_test_dir, exist_ok=True)

    def split_and_scale(self, target_column="Survived", test_size=0.2, random_state=42):
        logger.info("Reading ingested CSV for transformation")
        df = pd.read_csv(self.input_path)

        # Drop rows where target is NaN
        if df[target_column].isnull().sum() > 0:
            logger.warning(f"Dropping {df[target_column].isnull().sum()} rows with NaN in target column '{target_column}'")
            df = df.dropna(subset=[target_column]).reset_index(drop=True)

        logger.info("Splitting features and target")
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify numeric and categorical columns
        numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")

        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore")
)

        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols)
            ]
        )

        logger.info("Applying preprocessing to features")
        X_transformed = preprocessor.fit_transform(X)

        # Align transformed features with target
        cat_cols_encoded = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
        all_cols = list(numeric_cols) + list(cat_cols_encoded)
        X_transformed = pd.DataFrame(X_transformed, columns=all_cols)

        # Ensure same length after any preprocessing
        y = y.reset_index(drop=True)
        X_transformed = X_transformed.loc[y.index]

        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, test_size=test_size, random_state=random_state
        )

        # Save transformed data
        train_path = os.path.join(self.transformed_train_dir, "train.csv")
        test_path = os.path.join(self.transformed_test_dir, "test.csv")
        logger.info(f"Saving transformed train data to: {train_path}")
        pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
        logger.info(f"Saving transformed test data to: {test_path}")
        pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

        logger.info("Data transformation completed successfully")
        return train_path, test_path
