from tsai.all import *
import numpy as np
import pandas as pd
import os
import argparse
import datetime

def get_parser():
    description = "Times Series AI (TSAI) for Air Quality Sensing"
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
    df = pd.read_csv(os.path.join(args.data_folder, 'matrix_completion_preprocessed.csv'), dtype={'station_id': str})

    y = df['PM25_AQI_value']
    X = df.drop('PM25_AQI_value', axis=1)
    X = X.drop('station_id', axis=1)
    X = X.drop('time', axis=1)

    