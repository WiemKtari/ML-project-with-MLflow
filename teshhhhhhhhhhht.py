# debug_model.py
import joblib
import pandas as pd
import numpy as np

def debug_model(model_path="artifacts/model_trainer/model.joblib"):
    print("="*60)
    print("DEBUGGING MODEL STRUCTURE")
    print("="*60)
    
    # Load model
    model = joblib.load(model_path)
    
    # 1. Check pipeline structure
    print("\n1. PIPELINE STRUCTURE:")
    print(f"Pipeline steps: {list(model.named_steps.keys())}")
    
    # 2. Check preprocessor
    if 'preprocessor' in model.named_steps:
        preprocessor = model.named_steps['preprocessor']
        print(f"\n2. PREPROCESSOR TYPE: {type(preprocessor).__name__}")
        
        # Check transformers
        if hasattr(preprocessor, 'transformers'):
            print(f"\n3. TRANSFORMERS:")
            for i, (name, transformer, features) in enumerate(preprocessor.transformers):
                print(f"   Transformer {i}: {name}")
                print(f"   Features: {features}")
                print(f"   Transformer type: {type(transformer).__name__}")
                
                # If it's a pipeline
                if hasattr(transformer, 'named_steps'):
                    print(f"   Pipeline steps: {list(transformer.named_steps.keys())}")
                    
                    # Check for OneHotEncoder
                    if 'encoder' in transformer.named_steps:
                        encoder = transformer.named_steps['encoder']
                        print(f"   Encoder type: {type(encoder).__name__}")
                        
                        # Try to get categories
                        try:
                            if hasattr(encoder, 'categories_'):
                                categories = encoder.categories_
                                print(f"   Number of categories per feature: {[len(c) for c in categories]}")
                                if len(categories) > 0 and len(categories[0]) > 0:
                                    print(f"   First 5 categories of first feature: {categories[0][:5]}")
                            elif hasattr(encoder, 'categories'):
                                categories = encoder.categories
                                print(f"   Number of categories per feature: {[len(c) for c in categories]}")
                                if len(categories) > 0 and len(categories[0]) > 0:
                                    print(f"   First 5 categories of first feature: {categories[0][:5]}")
                        except:
                            print("   Could not get categories")
                
                print(f"   {'-'*40}")
    
    # 3. Test prediction with sample data
    print("\n4. TEST PREDICTION:")
    sample_data = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 25.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 50.0,
        'Embarked': 'S'
    }
    
    # Create DataFrame
    df = pd.DataFrame([sample_data])
    
    try:
        prediction = model.predict(df)
        print(f"   Sample prediction: {prediction[0]}")
        print("   ✅ Prediction successful!")
    except Exception as e:
        print(f"   ❌ Prediction failed: {str(e)}")
        
        # Try to understand what features the model expects
        print(f"\n5. ANALYZING INPUT ERROR:")
        
        # Check if model has feature_names_in_ attribute
        if hasattr(model, 'feature_names_in_'):
            print(f"   Model expects features: {model.feature_names_in_}")
        
        # Try to get feature names from preprocessor
        try:
            feature_names = preprocessor.get_feature_names_out()
            print(f"   Transformed feature names (first 20): {feature_names[:20]}")
            print(f"   Total transformed features: {len(feature_names)}")
        except:
            print("   Could not get transformed feature names")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    debug_model()