# src/mlProject/pipeline/prediction.py
import joblib
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self, model_path="artifacts/model_trainer/model.joblib"):
        """
        Load trained model and feature info from the specified path.
        """
        self.model_path = Path(model_path)
        self.feature_info_path = self.model_path.parent / "feature_info.joblib"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model does not exist at {self.model_path}")
        
        # Load model
        self.model = joblib.load(self.model_path)
        
        # Load feature info if available
        if self.feature_info_path.exists():
            self.feature_info = joblib.load(self.feature_info_path)
        else:
            # Default features if feature_info is not saved
            self.feature_info = {
                'all_features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            }
    
    def prepare_input(self, input_data: dict):
        """
        Prepare input data for prediction by:
        1. Removing non-predictive columns
        2. Converting to DataFrame
        3. Ensuring correct data types
        """
        # Non-predictive columns to remove
        columns_to_remove = ['Name', 'PassengerId', 'Ticket', 'Cabin']
        
        # Create a dictionary with only predictive features
        prediction_data = {}
        for feature in self.feature_info['all_features']:
            if feature in input_data:
                prediction_data[feature] = input_data[feature]
            else:
                # Handle missing features (use defaults or raise error)
                if feature == 'Pclass':
                    prediction_data[feature] = 3  # Default 3rd class
                elif feature == 'Sex':
                    prediction_data[feature] = 'male'
                elif feature == 'Age':
                    prediction_data[feature] = 30.0
                elif feature == 'SibSp':
                    prediction_data[feature] = 0
                elif feature == 'Parch':
                    prediction_data[feature] = 0
                elif feature == 'Fare':
                    prediction_data[feature] = 15.0
                elif feature == 'Embarked':
                    prediction_data[feature] = 'S'
        
        # Convert to DataFrame
        df = pd.DataFrame([prediction_data])
        
        # Ensure categorical columns are strings
        categorical_features = [col for col in ['Sex', 'Embarked'] if col in df.columns]
        for col in categorical_features:
            df[col] = df[col].astype(str)
        
        return df

    def predict(self, input_data: dict):
        """
        input_data: dict with all features from form
        Expected keys: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
        """
        # Prepare input data
        df = self.prepare_input(input_data)
        
        # Predict using the pipeline
        prediction = self.model.predict(df)
        
        # Return prediction (0 = did not survive, 1 = survived)
        return int(prediction[0])


# For testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "PassengerId": 1000,
        "Pclass": 1,
        "Name": "Test, Mr. John",
        "Sex": "male",
        "Age": 35.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "TEST123",
        "Fare": 50.0,
        "Cabin": "",
        "Embarked": "S"
    }
    
    try:
        predictor = PredictionPipeline()
        result = predictor.predict(sample_data)
        print(f"Prediction: {result} ({'Survived' if result == 1 else 'Did not survive'})")
    except Exception as e:
        print(f"Error: {e}")