# Formula 1 Race Winner Prediction

## Overview
This project uses machine learning to predict the winner of Formula 1 races based on historical data. The project includes an API, a web interface, and a machine learning model.

## Project Structure
- `/src`: Contains all source code.
  - `/api`: The Flask API for serving predictions.
  - `/models`: The machine learning models.
  - `/utils`: Helper functions for data processing.
  - `/tests`: Unit tests for the project.
- `/data`: Contains all dataset files.
- `/images`: Contains images for tracks and drivers.
- `/templates`: Contains the HTML files for the web interface.

## Setup Instructions
1. Clone this repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
1. Run the Flask API:
   ```bash
   python src/api/app.py
   ```
2. Access the web interface at `http://localhost:5000`.

## API Endpoints
- `/predict`: Returns a predicted winner based on input data.
