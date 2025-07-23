import pandas as pd
import joblib

# Load trained model and feature list
model = joblib.load("lungs_model.pkl")
feature_columns = joblib.load("lungs_model_features.pkl")

# Sample input values (replace these with your actual data)
input_data = {
    'Age': [50],
    'Gender': [1],  # assuming 1 = Male, 0 = Female (depends on your data)
    'Air Pollution': [2],
    'Alcohol use': [1],
    'Dust Allergy': [2],
    'OccuPational Hazards': [1],
    'Genetic Risk': [1],
    'chronic Lung Disease': [0],
    'Balanced Diet': [1],
    'Obesity': [0],
    'Smoking': [1],
    'Passive Smoker': [0],
    'Chest Pain': [1],
    'Coughing of Blood': [0],
    'Fatigue': [1],
    'Weight Loss': [0],
    'Shortness of Breath': [1],
    'Wheezing': [0],
    'Swallowing Difficulty': [1],
    'Clubbing of Finger Nails': [0],
    'Frequent Cold': [1],
    'Dry Cough': [1],
    'Snoring': [0]
}

# Create DataFrame
df = pd.DataFrame(input_data)

# Reorder columns and fill missing ones (if any)
for col in feature_columns:
    if col not in df.columns:
        df[col] = 0  # default/fallback value
df = df[feature_columns]  # make sure order matches exactly

# Predict
prediction = model.predict(df)
print("Predicted Class:", prediction)
