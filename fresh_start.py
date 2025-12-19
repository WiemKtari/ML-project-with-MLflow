# fresh_start.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def create_fresh_pipeline():
    print("CREATING FRESH TITANIC PIPELINE")
    print("="*60)
    
    # Step 1: Download or load raw Titanic data
    try:
        # Try to load from your current data to see structure
        current_data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        print(f"Loaded raw Titanic data: {current_data.shape}")
        print(f"Columns: {list(current_data.columns)}")
        
        # Use this raw data
        df = current_data
    except:
        # Create synthetic data if download fails
        print("Creating synthetic Titanic-like data...")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.randint(0, 2, n_samples),
            'Pclass': np.random.choice([1, 2, 3], n_samples),
            'Name': [f'Passenger_{i}' for i in range(n_samples)],
            'Sex': np.random.choice(['male', 'female'], n_samples),
            'Age': np.random.uniform(1, 80, n_samples),
            'SibSp': np.random.randint(0, 5, n_samples),
            'Parch': np.random.randint(0, 4, n_samples),
            'Ticket': [f'Ticket_{i}' for i in range(n_samples)],
            'Fare': np.random.uniform(5, 200, n_samples),
            'Cabin': [f'Cabin_{np.random.randint(1, 50)}' for _ in range(n_samples)],
            'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples)
        })
    
    # Step 2: Basic preprocessing
    print("\nPreprocessing data...")
    
    # Handle missing values
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna('S')
    
    # Step 3: Define features and target
    target_column = 'Survived'
    
    # Remove non-predictive columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=[target_column] + columns_to_drop)
    y = df[target_column]
    
    print(f"\nFeatures to use: {list(X.columns)}")
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 5: Create preprocessing pipeline
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Step 6: Create and train model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
    ])
    
    print("\nTraining model...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = model_pipeline.score(X_train, y_train)
    test_score = model_pipeline.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Step 7: Save everything
    artifacts_dir = "artifacts"
    os.makedirs(f"{artifacts_dir}/data_transformation/train", exist_ok=True)
    os.makedirs(f"{artifacts_dir}/data_transformation/test", exist_ok=True)
    os.makedirs(f"{artifacts_dir}/model_trainer", exist_ok=True)
    
    # Save train/test data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(f"{artifacts_dir}/data_transformation/train/train.csv", index=False)
    test_data.to_csv(f"{artifacts_dir}/data_transformation/test/test.csv", index=False)
    
    # Save model
    model_path = f"{artifacts_dir}/model_trainer/model.joblib"
    joblib.dump(model_pipeline, model_path)
    
    # Save feature info
    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': numeric_features + categorical_features
    }
    joblib.dump(feature_info, f"{artifacts_dir}/model_trainer/feature_info.joblib")
    
    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Feature info saved")
    print(f"✅ Train data: {train_data.shape}")
    print(f"✅ Test data: {test_data.shape}")
    
    # Step 8: Test prediction
    print("\nTesting prediction...")
    sample_data = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 25.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 50.0,
        'Embarked': 'S'
    }
    
    sample_df = pd.DataFrame([sample_data])
    
    # Ensure categorical columns are strings
    for col in categorical_features:
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].astype(str)
    
    try:
        prediction = model_pipeline.predict(sample_df)
        probability = model_pipeline.predict_proba(sample_df)
        
        print(f"✅ Prediction: {prediction[0]} ({'Survived' if prediction[0] == 1 else 'Did not survive'})")
        print(f"✅ Probability: {probability[0]}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    return model_pipeline

if __name__ == "__main__":
    create_fresh_pipeline()