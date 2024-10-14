import pandas as pd

# Load the datasets
drivers_df = pd.read_csv('data/drivers.csv')
results_df = pd.read_csv('data/results.csv')
races_df = pd.read_csv('data/races.csv')
constructors_df = pd.read_csv('data/constructors.csv')

# List of drivers you're interested in
filtered_drivers = [
    "Max Verstappen", "Sergio Pérez", "Lewis Hamilton", "George Russell", 
    "Charles Leclerc", "Carlos Sainz Jr.", "Lando Norris", "Oscar Piastri", 
    "Fernando Alonso", "Lance Stroll", "Esteban Ocon", "Pierre Gasly", 
    "Valtteri Bottas", "Guanyu Zhou", "Kevin Magnussen", "Nico Hülkenberg", 
    "Yuki Tsunoda", "Daniel Ricciardo", "Alex Albon", "Logan Sargeant"
]

# Combine forename and surname to match full names
drivers_df['full_name'] = drivers_df['forename'] + " " + drivers_df['surname']
filtered_drivers_df = drivers_df[drivers_df['full_name'].isin(filtered_drivers)]

# Filter results for the selected drivers
filtered_results_with_races = results_df[results_df['driverId'].isin(filtered_drivers_df['driverId'])]
filtered_results_with_races = filtered_results_with_races.merge(races_df, on='raceId', how='left')
filtered_results_with_races = filtered_results_with_races.merge(constructors_df, on='constructorId', how='left')

# Prepare the input features and target
X_filtered = filtered_results_with_races[['grid', 'driverId', 'constructorId', 'raceId']]
y_filtered = filtered_results_with_races['positionOrder']

# Save the filtered data to use for model training
X_filtered.to_csv('data/X_filtered.csv', index=False)
y_filtered.to_csv('data/y_filtered.csv', index=False)
print("Data prepared and saved.")
