import numpy as np
import pandas as pd
import torch
import math
from torch import nn
import torch.nn.functional as F
import argparse
import os
import datetime
import sys
import tracemalloc

def get_parser():
    description = "MiniRocket with RandomForest Classifier and Cross Validation"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_folder', type=str, help="Path to the output folder")
    parser.add_argument('-t', '--training_folder', type=str, help="Path to the training folder")
    parser.add_argument('-v', '--validation_folder', type=str, help="Path to the validation folder")
    return parser

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

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

class SpatiotemporalEnconder(nn.Module):
    def __init__(self, model_dim):
        super(SpatiotemporalEnconder, self).__init__()
        self.model_dim = model_dim
        self.position_embedding_layer = nn.Linear(3, model_dim)
        self.time_embedding_layer = nn.Linear(8, model_dim)
        self.numerical_embedding_layer = nn.Linear(7, model_dim)

    def forward(self, X):
        if 'longitude' in X.columns and 'latitude' in X.columns:
            x, y, z = encode_coordinates(X['latitude'], X['longitude'])
            x = torch.tensor(x.values, dtype=torch.float32)
            y = torch.tensor(y.values, dtype=torch.float32)
            z = torch.tensor(z.values, dtype=torch.float32)
            
            position_embedding = self.position_embedding_layer(torch.stack([x, y, z], dim=-1))
        else:
            raise ValueError("Input data must contain 'latitude' and 'longitude' columns")
    
        if 'month' in X.columns and 'day' in X.columns and 'hour' in X.columns and 'year' in X.columns and 'AM_PM' in X.columns:
            month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, year, AM_PM = encode_time(X['month'], X['day'], X['hour'], X['year'], X['AM_PM'])
            month_sin = torch.tensor(month_sin.values, dtype=torch.float32)
            month_cos = torch.tensor(month_cos.values, dtype=torch.float32)
            day_sin = torch.tensor(day_sin.values, dtype=torch.float32)
            day_cos = torch.tensor(day_cos.values, dtype=torch.float32)
            hour_sin = torch.tensor(hour_sin.values, dtype=torch.float32)
            hour_cos = torch.tensor(hour_cos.values, dtype=torch.float32)
            year = torch.tensor(year.values, dtype=torch.float32)
            AM_PM = torch.tensor(AM_PM.values, dtype=torch.float32)

            time_embedding = self.time_embedding_layer(torch.stack([month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos, year, AM_PM], dim=-1))
        else:
            raise ValueError("Input data must contain 'month', 'day', 'hour', 'year', and 'AM_PM' columns")
        
        X_remaining = X.copy()
        X_remaining.drop(columns=['latitude', 'longitude', 'month', 'day', 'hour', 'year', 'AM_PM', 'station_id'], inplace=True)

        if X_remaining.shape[1] == 7:
            numerical_embedding = self.numerical_embedding_layer(torch.tensor(X_remaining.values, dtype=torch.float32))
        else:
            raise ValueError("Incorrect Numerical Features")
        
        embeddings = position_embedding + time_embedding + numerical_embedding
        batch_size = X.shape[0]
        sequence_length = 1 
        embeddings = embeddings.view(batch_size, sequence_length, self.model_dim)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers):
        super().__init__()
        self.spatiotemporal_embedding = SpatiotemporalEnconder(d_model)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask):
        x = self.spatiotemporal_embedding(x)
        x = x.clone().detach().requires_grad_(False)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention, removing in outer function!
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.linear = nn.Linear(d_model, 1)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask)
        out = self.linear(x)
        return out

output_folder = 'output'
start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'log_{start_time}'

def log(message):
    with open(f'{os.path.join(output_folder, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

if __name__ == "__main__":
    tracemalloc.start()

    parser = get_parser()
    args = parser.parse_args()
    device = get_device()
    log(device)

    output_folder = args.output_folder

    log('Loading data')
    # Load the data
    train_numerical_features = pd.read_csv(os.path.join(args.training_folder, 'Features.csv'))
    train_position_features = pd.read_csv(os.path.join(args.training_folder, 'PositionFeatures.csv'))
    train_time_features = pd.read_csv(os.path.join(args.training_folder, 'TimeFeatures.csv'))
    train_values = pd.read_csv(os.path.join(args.training_folder, 'Values.csv'))

    X_train = pd.concat([train_numerical_features, train_position_features, train_time_features], axis=1)
    y_train = train_values.drop(columns=['station_id'])

    val_numerical_features = pd.read_csv(os.path.join(args.validation_folder, 'Features.csv'))
    val_position_features = pd.read_csv(os.path.join(args.validation_folder, 'PositionFeatures.csv'))
    val_time_features = pd.read_csv(os.path.join(args.validation_folder, 'TimeFeatures.csv'))
    val_values = pd.read_csv(os.path.join(args.validation_folder, 'Values.csv'))

    X_val = pd.concat([val_numerical_features, val_position_features, val_time_features], axis=1)
    y_val = val_values.drop(columns=['station_id'])

    # Sample 1000 from train and 100 from val
    X_train = X_train.sample(n=10000, random_state=42)
    y_train = y_train.loc[X_train.index]
    X_val = X_val.sample(n=10000, random_state=42)
    y_val = y_val.loc[X_val.index]

    # Define the model
    model = Transformer(d_model=128, ffn_hidden=512, num_heads=8, drop_prob=0.001, num_layers=96)

    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    X_train = X_train.fillna(0)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    y_train = y_train.clone().detach().requires_grad_(False)
    y_train[torch.isnan(y_train)] = 0

    log('Training model')
    # Train the model
    for epoch in range(5):
        optimizer.zero_grad()
        y_pred = model(X_train, y_train)
        y_pred = y_pred.view(-1, 1)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        log(f'Epoch {epoch+1} Loss {loss.item()}')

    # Test the model
    X_val = X_val.fillna(0)
    y_val = y_val.fillna(0)
    y_pred = model(X_val, y_val)
    y_pred = y_pred.view(-1, 1)
    loss = criterion(y_pred, torch.tensor(y_val.values, dtype=torch.float32))
    mean_val = torch.mean(torch.tensor(y_val.values, dtype=torch.float32)).item()
    percent_error = torch.sqrt(loss).item() / mean_val
    log(f'Validation Loss {loss.item()}')
    log(f'Validation % Error {percent_error}')
    log(f'Validation RMSE {torch.sqrt(loss).item()}')


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    memory = params * 4 / (1024 ** 2)  # Assuming 4 bytes per parameter (float32)
    log(f'Model memory usage: {memory:.2f} MB')

    current, peak = tracemalloc.get_traced_memory()
    log(f'Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB')
    tracemalloc.stop()

