import pandas as pd
import joblib
import os

def create_submission():
    print("Loading data...")
    # Typically Kaggle provides a test.csv without labels. 
    # Here we'll use Iris.csv and simulate it by dropping the label.
    if os.path.exists('test.csv'):
        df = pd.read_csv('test.csv')
    elif os.path.exists('Iris.csv'):
        print("Using Iris.csv as test data (simulated).")
        df = pd.read_csv('Iris.csv')
    else:
        print("No data found!")
        return

    # Features needed
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    # Ensure ID exists
    if 'Id' not in df.columns:
        df['Id'] = range(1, len(df) + 1)
        
    X_test = df[features]
    
    print("Loading model...")
    # Load the best model (Random Forest)
    try:
        model = joblib.load('models/random_forest.pkl')
    except FileNotFoundError:
        print("Model not found. Please train models first.")
        return

    print("Predicting...")
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'Id': df['Id'],
        'Species': predictions
    })
    
    output_file = 'submission.csv'
    submission.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")
    print(submission.head())

if __name__ == "__main__":
    create_submission()
