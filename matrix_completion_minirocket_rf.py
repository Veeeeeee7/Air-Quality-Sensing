import numpy as np
import pandas as pd
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sktime.utils import mlflow_sktime 
from sklearn.ensemble import RandomForestRegressor
import argparse
import datetime
from sklearn.metrics import root_mean_squared_error
import os

def get_parser():
    description = "MiniRocket with RandomForest Regressor"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_file', type=str, help="Path to the output file")
    parser.add_argument('-t', '--training_file', type=str, help="Path to the training file")
    parser.add_argument('-v', '--validation_file', type=str, help="Path to the validation file")
    return parser

def encode_coordinates(latitude, longitude):
    # Convert degrees to radians using numpy
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z

def encode_time(month, day, hour, year, AM_PM):
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    year = year.astype(float)
    AM_PM = AM_PM.astype(float)
    return month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, year, AM_PM

def process_data(X):
    if 'longitude' in X.columns and 'latitude' in X.columns:
        x, y, z = encode_coordinates(X['latitude'], X['longitude'])
    else:
        raise ValueError("Input data must contain 'latitude' and 'longitude' columns")
    
    if 'month' in X.columns and 'day' in X.columns and 'hour' in X.columns and 'year' in X.columns and 'AM_PM' in X.columns:
        month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, year, AM_PM = encode_time(X['month'], X['day'], X['hour'], X['year'], X['AM_PM'])
    else:
        raise ValueError("Input data must contain 'month', 'day', 'hour', 'year', and 'AM_PM' columns")
    
    processed_data = pd.DataFrame({'x': x, 'y': y, 'z': z, 'month_sin': month_sin, 'month_cos': month_cos, 'day_sin': day_sin, 'day_cos': day_cos, 'hour_sin': hour_sin, 'hour_cos': hour_cos, 'year': year, 'AM_PM': AM_PM})

    X.drop(columns=['longitude', 'latitude', 'month', 'day', 'hour', 'year', 'AM_PM', 'station_id'], inplace=True)
    return pd.concat([X, processed_data], axis=1)

output_file = 'output'
start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'log_{start_time}'

def log(message):
    with open(f'{os.path.join(output_file, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    output_file = args.output_file

    log("Loading data")
    train_df = pd.read_csv(args.training_file)
    val_df = pd.read_csv(args.validation_file)

    X_train = train_df.drop(columns=['PM25_AQI_value'])
    y_train = train_df['PM25_AQI_value']

    X_val = val_df.drop(columns=['PM25_AQI_value'])
    y_val = val_df['PM25_AQI_value']

    X_train_sampled = X_train[:1000]
    y_train_sampled = y_train[:1000]

    print(X_train_sampled.columns)

    X_train_processed = process_data(X_train_sampled)
    X_val_processed = process_data(X_val)

    log("Processing data with MiniRocket")
    minirocket = MiniRocketMultivariate()
    X_train_transformed = minirocket.fit_transform(X_train_processed)
    X_val_transformed = minirocket.transform(X_val_processed)

    log("Training RandomForest Regressor")
    n_estimators=100
    max_depth=10
    min_samples_split=4
    min_samples_leaf=2
    max_features='sqrt'
    oob_score=True
    criterion='squared_error'
    random_state=42

    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, oob_score=oob_score, criterion=criterion, random_state=random_state)
    rf.fit(X_train_transformed, y_train_sampled)
    
    log("Evaluating model")
    y_pred = rf.predict(X_val_transformed)

    log('Calculating training RMSE from OOB predictions')
    # Calculate training RMSE from OOB predictions
    oob_predictions = rf.oob_prediction_
    oob_rmse = np.sqrt(np.mean((y_train_sampled - oob_predictions) ** 2))
    log(f'Training RMSE (OOB): {oob_rmse}')

    log('Calculating RMSE')
    # Calculate RMSE
    rmse = root_mean_squared_error(y_val, y_pred)
    log(f'RMSE: {rmse}')