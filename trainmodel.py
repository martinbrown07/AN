import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your dataset
df = pd.read_csv("anemia data from Kaggle.csv")  # Replace with your actual CSV file

# Extract features and labels (adjust these column names)
X = df[['Gender', 'Hemoglobin', 'MCV']]  # Use the actual columns you used to train
y = df['Result']  # Binary output: 1 for Anemic, 0 for Non-Anemic

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model using pickle
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'random_forest_model.pkl'")
