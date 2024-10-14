import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Step 2: Filter the dataset for specific drivers
drivers = [
    'Max Verstappen', 'Sergio Pérez', 'Lewis Hamilton', 'George Russell', 
    'Charles Leclerc', 'Carlos Sainz Jr.', 'Lando Norris', 'Oscar Piastri', 
    'Fernando Alonso', 'Lance Stroll', 'Esteban Ocon', 'Pierre Gasly', 
    'Valtteri Bottas', 'Guanyu Zhou', 'Kevin Magnussen', 'Nico Hülkenberg', 
    'Yuki Tsunoda', 'Daniel Ricciardo', 'Alex Albon', 'Logan Sargeant'
]
df_filtered = df[df['Driver'].isin(drivers)]

# Step 3: Split the dataset into features and target variables
X = df_filtered.drop(['Position', 'Driver'], axis=1)
y = df_filtered['Position']

# Step 4: Train-test split
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Define the parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Step 7: Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Step 8: Initialize RandomizedSearchCV for hyperparameter tuning
rf_random = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_grid, 
    n_iter=100, 
    cv=5, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1
)

# Step 9: Fit the model to the training data
rf_random.fit(X_train_filtered, y_train_filtered)

# Step 10: Predict and calculate RMSE
y_pred_filtered = rf_random.predict(X_test_filtered)
rmse = mean_squared_error(y_test_filtered, y_pred_filtered, squared=False)
print(f"RMSE after tuning: {rmse}")
