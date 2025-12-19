# src/mlProject/components/model_trainer.py
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import joblib
from mlProject.logger import logger

class ModelTrainer:
    def __init__(self, train_path, test_path, model_params=None, target_column="Survived", save_model_path=None):
        """
        train_path: path to the training CSV
        test_path: path to the test CSV
        model_params: dict of hyperparameters for RandomForestClassifier
        target_column: name of the target column
        save_model_path: path to save the trained model (optional)
        """
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = target_column
        self.model_params = model_params if model_params is not None else {}
        self.save_model_path = save_model_path

    def preprocess_data(self, df):
        """
        Preprocess the data: remove non-predictive columns and handle missing values
        """
        # Remove non-predictive columns
        columns_to_drop = ['Name', 'PassengerId', 'Ticket', 'Cabin']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Convert categorical columns to string type
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            
        return df

    def train_model(self):
        # Load data
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # Preprocess data
        train_df = self.preprocess_data(train_df)
        test_df = self.preprocess_data(test_df)

        # Separate features and target
        X_train = train_df.drop(columns=[self.target_column])
        y_train = train_df[self.target_column]

        X_test = test_df.drop(columns=[self.target_column])
        y_test = test_df[self.target_column]

        # Feature columns after preprocessing
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")

        # Preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Complete pipeline with RandomForest
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**self.model_params))
        ])

        # Enable MLflow autologging
        mlflow.sklearn.autolog()

        # Train the model
        logger.info("Training RandomForest model...")
        model_pipeline.fit(X_train, y_train)
        logger.info("Model trained successfully!")

        # Quick evaluation
        accuracy = model_pipeline.score(X_test, y_test)
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)

        # Save feature information for prediction pipeline
        feature_info = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'all_features': numeric_features + categorical_features
        }
        
        # Optional: save the trained model locally
        if self.save_model_path:
            os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
            
            # Save model pipeline
            joblib.dump(model_pipeline, self.save_model_path)
            
            # Save feature info separately
            feature_info_path = os.path.join(os.path.dirname(self.save_model_path), 'feature_info.joblib')
            joblib.dump(feature_info, feature_info_path)
            
            logger.info(f"Model saved at {self.save_model_path}")
            logger.info(f"Feature info saved at {feature_info_path}")

        return model_pipeline, feature_info


# Example usage:
if __name__ == "__main__":
    best_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    }

    trainer = ModelTrainer(
        train_path="artifacts/data_transformation/train/train.csv",
        test_path="artifacts/data_transformation/test/test.csv",
        model_params=best_params,
        save_model_path="artifacts/model_trainer/model.joblib"
    )
    model, feature_info = trainer.train_model()
    print(f"Model trained. Features used: {feature_info['all_features']}")