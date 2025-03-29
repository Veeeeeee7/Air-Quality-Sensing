import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import datetime

def get_parser():
    description = "MiniRocket with RandomForest Classifier and Cross Validation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_folder', type=str, help="Path to the output folder")
    parser.add_argument('-t', '--training_folder', type=str, help="Path to the training folder")
    parser.add_argument('-v', '--validation_folder', type=str, help="Path to the validation folder")
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

output_folder = 'output'
start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'log_{start_time}'

def log(message):
    with open(f'{os.path.join(output_folder, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

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

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    output_folder = args.output_folder

    log('Loading data')
    # Load the data
    train_numerical_features = pd.read_csv(os.path.join(args.training_folder, 'Features.csv'))
    train_position_features = pd.read_csv(os.path.join(args.training_folder, 'PositionFeatures.csv'))
    train_time_features = pd.read_csv(os.path.join(args.training_folder, 'TimeFeatures.csv'))
    train_values = pd.read_csv(os.path.join(args.training_folder, 'Values.csv'))

    X_train = process_data(pd.concat([train_numerical_features, train_position_features, train_time_features], axis=1))
    y_train = train_values.drop(columns=['station_id'])
    X_train = X_train.fillna(-1)
    y_train = y_train.fillna(-1)

    # Sample 1000 from X_train and y_train
    sample_indices = np.random.choice(X_train.index, size=1000, replace=False)
    X_train_sampled = X_train.loc[sample_indices]
    y_train_sampled = y_train.loc[sample_indices]
    
    X_train_sampled = X_train
    y_train_sampled = y_train
    y_train_sampled = y_train_sampled.values.ravel()

    log('Training model')
    # Train the model
    n_estimators=100
    max_depth=10
    learning_rate=0.1
    subsample=0.8
    colsample_bytree=0.8
    random_state=42
    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree, random_state=random_state)
    model.fit(X_train_sampled, y_train_sampled)

    log('Loading validation data')
    # Load the validation data
    validation_numerical_features = pd.read_csv(os.path.join(args.validation_folder, 'Features.csv'))
    validation_position_features = pd.read_csv(os.path.join(args.validation_folder, 'PositionFeatures.csv'))
    validation_time_features = pd.read_csv(os.path.join(args.validation_folder, 'TimeFeatures.csv'))
    validation_values = pd.read_csv(os.path.join(args.validation_folder, 'Values.csv'))

    X_validation = process_data(pd.concat([validation_numerical_features, validation_position_features, validation_time_features], axis=1))
    y_validation = validation_values.drop(columns=['station_id'])
    X_validation = X_validation.fillna(-1)
    y_validation = y_validation.fillna(-1)
    y_validation = y_validation.values.ravel()

    log('Predicting')
    # Predict
    y_pred = model.predict(X_validation)

    log('Calculating training RMSE')
    # Calculate training RMSE
    y_train_pred = model.predict(X_train_sampled)
    train_rmse = root_mean_squared_error(y_train_sampled, y_train_pred)
    log(f'Training RMSE: {train_rmse}')

    log('Calculating RMSE')
    # Calculate RMSE
    rmse = root_mean_squared_error(y_validation, y_pred)
    log(f'RMSE: {rmse}')

    


    