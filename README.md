# BMI Prediction API

This project implements a simple Flask API for predicting BMI (Body Mass Index) categories based on input data. The API uses a RandomForestClassifier model trained on a dataset containing BMI information.

## Files in the Project

- **app.py**: Contains the Flask API implementation with two endpoints, one for testing the API (`/`) and another for getting the BMI prediction output (`/getPredictionOutput`).
- **model.py**: Contains the code for loading the dataset, preprocessing the data, training the RandomForestClassifier model using Grid Search for hyperparameter tuning, and saving the best model to a file.
- **prediction.py**: Contains a function for loading the trained model and making predictions based on input data.

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask API using `python app.py`. The API will be available at `http://localhost:5000`.
4. Use a tool like Postman or curl to test the API endpoints. You can send a POST request to `/getPredictionOutput` with JSON data containing the features required for prediction.

## Example Request

```json
{
    "Gender": 1,
    "Height": 170,
    "Weight": 70,
    "Age": 25
}
```

## Example Response

```json
{
    "predict": "Normal"
}
```
