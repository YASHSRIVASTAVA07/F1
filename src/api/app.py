from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__,
            template_folder='C:/Users/yashs/Desktop/F1_project_python_312_complete/src/api/templates',
            static_folder='C:/Users/yashs/Desktop/F1_project_python_312_complete/static')

# Path to the dataset directory
dataset_path = 'C:/Users/yashs/Desktop/F1_project_python_312_complete/data/'

@app.route('/')
def home():
    # Load the driver and track data
    drivers_df = pd.read_csv(os.path.join(dataset_path, 'drivers.csv'))
    drivers_df['driver_full_name'] = drivers_df['forename'] + ' ' + drivers_df['surname']
    drivers_list = drivers_df['driver_full_name'].tolist()

    tracks_df = pd.read_csv(os.path.join(dataset_path, 'races.csv'))
    tracks_list = tracks_df['name'].tolist()

    # Pass the data to the template
    return render_template('index.html', drivers=drivers_list, tracks=tracks_list)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extracting data from the form
    driver_forename = data.get('forename')
    driver_surname = data.get('surname')
    circuit = data.get('circuit')
    avg_position = data.get('avg_position')

    # Load the datasets
    drivers_df = pd.read_csv(os.path.join(dataset_path, 'drivers.csv'))
    races_df = pd.read_csv(os.path.join(dataset_path, 'races.csv'))
    results_df = pd.read_csv(os.path.join(dataset_path, 'results.csv'))

    # Find the driver ID based on the given name
    driver_id = drivers_df[(drivers_df['forename'] == driver_forename) &
                           (drivers_df['surname'] == driver_surname)]['driverId'].values[0]

    # Use race and result data to compute statistics for the driver
    driver_results = results_df[results_df['driverId'] == driver_id]

    # Extract specific race data based on the circuit
    race_id = races_df[races_df['name'] == circuit]['raceId'].values
    if len(race_id) > 0:
        race_results = driver_results[driver_results['raceId'] == race_id[0]]
        if len(race_results) > 0:
            predicted_position = race_results['positionOrder'].mean()  # Average position on this track
        else:
            predicted_position = driver_results['positionOrder'].mean()  # General average
    else:
        predicted_position = driver_results['positionOrder'].mean()  # Fallback to general average

    # Introduce some advanced logic to adjust predictions based on recent performance and track history
    if avg_position > 0:
        predicted_position = (predicted_position + avg_position) / 2

    # Round and convert the predicted position to an integer for final prediction
    predicted_position = int(round(predicted_position))

    return jsonify({'prediction': predicted_position})

if __name__ == '__main__':
    app.run(debug=True)
