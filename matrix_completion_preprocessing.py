import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/AirQualityBeijing/Beijing/CrawledData.txt", sep=",", dtype={'station_id': str})
stations = {station_id: station_df for station_id, station_df in df.groupby('station_id')}
processed = []
for station_id, station_df in stations.items():
    print(f"Processing station: {station_id}")
    grouped = [station_df.iloc[i:i + 12] for i in range(0, len(station_df), 12)]
    for group in grouped:
        row = pd.DataFrame({col: [group[col].astype(np.float32) if group[col].dtype == np.float64 else group[col]] for col in group.columns})
        processed.append(row)

processed_df = pd.concat(processed)


train_df, temp = train_test_split(processed_df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)


train_df.to_csv("Data/MatrixCompletion/Train/matrix_completion_preprocessed_train.csv", index=False)
val_df.to_csv("Data/MatrixCompletion/Val/matrix_completion_preprocessed_val.csv", index=False)
test_df.to_csv("Data/MatrixCompletion/Test/matrix_completion_preprocessed_test.csv", index=False)