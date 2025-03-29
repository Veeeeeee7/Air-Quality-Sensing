import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import argparse
import datetime
import os

def get_parser():
    description = "Autogluon Time Series Predictor for Air Quality Sensing"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_folder', type=str, help="Path to the output folder")
    parser.add_argument('-d', '--data_folder', type=str, help="Path to the data folder")
    return parser

output_folder = 'output'
start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'log_{start_time}'

def log(message):
    with open(f'{os.path.join(output_folder, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    df_crawled = pd.read_csv(os.path.join(args.data_folder, 'CrawledData.txt'), sep=',', dtype={'station_id': str})
    df_station = pd.read_csv(os.path.join(args.data_folder, 'Station.txt'), sep=',', dtype={'station_id': str})
    df = pd.merge(df_crawled, df_station, on='station_id')
    df['PM25_AQI_value'] = (df['PM25_AQI_value'] - df['PM25_AQI_value'].mean()) / df['PM25_AQI_value'].std()
    train_data = TimeSeriesDataFrame(df, id_column='station_id', timestamp_column='time')

    predictor = TimeSeriesPredictor(path=os.path.join(args.output_folder, 'model'), prediction_length=12, target='PM25_AQI_value', eval_metric='RMSE', freq='H')
    predictor.fit(train_data, presets='best_quality', memory=100, time_limit=600)