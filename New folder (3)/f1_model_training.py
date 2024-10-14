# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Load the dataset CSVs
drivers = pd.read_csv(r'C:\Users\yashs\Desktop\New folder (3)\data\drivers.csv')
races = pd.read_csv(r'C:\Users\yashs\Desktop\New folder (3)\data\races.csv')
results = pd.read_csv(r'C:\Users\yashs\Desktop\New folder (3)\data\results.csv')
qualifying = pd.read_csv(r'C:\Users\yashs\Desktop\New folder (3)\data\qualifying.csv')
driver_standings = pd.read_csv(r'C:\Users\yashs\Desktop\New folder (3)\data\driver_standings.csv')
constructor_standings = pd.read_csv(r'C:\Users\yashs\Desktop\New folder (3)\data\constructor_standings.csv')

# Merge dataframes as per your logic to create a dataset for training
# (You need to adjust the merging based on the relations between these datasets.)
# This is a placeholder merge, you will adjust this based on your data structure.

data = pd.merge(results, races, on='raceId')
data = pd.merge(data, drivers, on='driverId')

# Data preprocessing (you need to adapt it to your data)
# For simplicity, let's take driver standings as a target and some features.
X = data[['positionOrder', 'grid', 'laps']]  # Adjust these columns based on your dataset
y = data['points']  # Assuming points is the target (You can change it to another target variable)

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor model with hyperparameters
rf = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=2, min_samples_leaf=2, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf, 'f1_rf_model.pkl')

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')
