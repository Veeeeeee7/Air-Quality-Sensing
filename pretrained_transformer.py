import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import pandas as pd
import os
import argparse
import datetime

def get_parser():
    description = "Pretrained Transformer Model for Spatial-Temporal Data"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', '--output_folder', type=str, help="Path to the output folder")
    parser.add_argument('-t', '--training_data', type=str, help="Path to the training data")
    parser.add_argument('-v', '--validation_data', type=str, help="Path to the validation data")
    parser.add_argument('-m', '--model_folder', type=str, help="Path to the model folder")
    return parser

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class SpatialEmbedding(nn.Module):
    def __init__(self, d_spatial, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_spatial = d_spatial
        self.embedding = nn.Linear(d_spatial, d_model)

    def forward(self, x):
        return self.embedding(x)

class TemporalEmbedding(nn.Module):
    def __init__(self, d_temporal, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_temporal = d_temporal
        self.embedding = nn.Linear(d_temporal, d_model)

    def forward(self, x):
        return self.embedding(x)

class ValueEmbedding(nn.Module):
    def __init__(self, d_value, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_value = d_value
        self.embedding = nn.Linear(d_value, d_model)
    
    def forward(self, x):
        return self.embedding(x)
    
class InputEmbedding(nn.Module):
    def __init__(self, d_model, d_spatial, d_temporal, d_value):
        super().__init__()
        self.d_model = d_model
        self.spatial_embedding = SpatialEmbedding(d_spatial, d_model)
        self.temporal_embedding = TemporalEmbedding(d_temporal, d_model)
        self.value_embedding = ValueEmbedding(d_value, d_model)

    def forward(self, x):
        x_spatial = x[spatial_columns]
        x_temporal = x[temporal_columns]
        x_value = x[value_columns]

        x_spatial = torch.tensor(x_spatial.values).to(get_device())
        x_temporal = torch.tensor(x_temporal.values).to(get_device())
        x_value = torch.tensor(x_value.values).to(get_device())

        spatial = self.spatial_embedding(x_spatial)
        temporal = self.temporal_embedding(x_temporal)
        value = self.value_embedding(x_value)
        return spatial + temporal + value

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v)
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
    def __init__(self, d_model, hidden, drop_prob=0.1):
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

    def forward(self, x):
        residual_x = x.clone()
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x = inputs[0]
        for module in self._modules.values():
            x = module(x)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model,
                 d_spatial,
                 d_temporal,
                 d_value, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers):
        super().__init__()
        self.embedding = InputEmbedding(d_model, d_spatial, d_temporal, d_value)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x):
        x_embedding = self.embedding(x)
        x_embedding = x_embedding.view(x_embedding.size(0), -1, self.embedding.d_model)
        output = self.layers(x_embedding)
        return output
    
class Decoder(nn.Module):
    def __init__(self, d_model, d_spatial, d_temporal, d_value):
        super().__init__()
        self.spatial_linear = nn.Linear(d_model, d_spatial)
        self.temporal_linear = nn.Linear(d_model, d_temporal)
        self.value_linear = nn.Linear(d_model, d_value)

    def forward(self, x):
        spatial = self.spatial_linear(x)
        temporal = self.temporal_linear(x)
        value = self.value_linear(x)
        return torch.cat((spatial, temporal, value), dim=-1).view(x.size(0), -1)

class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                d_spatial,
                d_temporal,
                d_value,
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                mask_prob, 
                num_layers):
        super().__init__()
        self.encoder = Encoder(d_model, d_spatial, d_temporal, d_value, ffn_hidden, num_heads, drop_prob, num_layers)
        self.decoder = Decoder(d_model, d_spatial, d_temporal, d_value)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        embedding = self.encoder(x)
        output = self.decoder(embedding)
        return output

class SpatialTemporalMask():
    def __init__(self, mask_prob):
        super().__init__()
        self.mask_prob = mask_prob

    def forward(self, x):
        masked_indices = np.random.rand(x.shape[0]) < self.mask_prob
        columns = x.columns
        for column in columns:
            x.loc[masked_indices, column] = np.float32(-1000)

        return x, masked_indices

def log(message):
    with open(f'{os.path.join(output_folder, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

output_folder = ''
start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'log_{start_time}'

spatial_columns = ['longitude', 'latitude']
temporal_columns = ['day', 'month', 'year', 'hour']
value_columns = ['PM25_Concentration', 'PM10_Concentration',
       'NO2_Concentration', 'CO_Concentration', 'O3_Concentration',
       'SO2_Concentration','temperature', 'pressure',
       'humidity', 'wind_speed', 'wind_direction', 'Cloudy', 'Dusty', 'Foggy',
       'Freezing rain', 'Heavier rain', 'Heavy snow', 'Light snow',
       'Moderate rain', 'Moderate snow', 'Overcast', 'Rain storm', 'Rainy',
       'Sand storm', 'Sprinkle', 'Sunny', 'Thunder storm']

if __name__ == '__main__':
    d_model = 128
    d_spatial = len(spatial_columns)
    d_temporal = len(temporal_columns)
    d_value = len(value_columns)
    ffn_hidden = 512
    num_heads = 8
    drop_prob = 0.1
    mask_prob = 0.15
    num_layers = 6
    batch_size = 32
    torch.set_default_dtype(torch.float32)

    args = get_parser().parse_args()
    training_data = args.training_data
    validation_data = args.validation_data
    output_folder = args.output_folder
    model_folder = args.model_folder

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    x_train = pd.read_csv(training_data, dtype=np.float32)
    x_train = x_train.drop(columns=['station_id'])
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    x_train = x_train.fillna(-1)


    x_val = pd.read_csv(validation_data, dtype=np.float32)
    x_val = x_val.drop(columns=['station_id'])
    x_val = (x_val - x_val.min()) / (x_val.max() - x_val.min())
    x_val = x_val.fillna(-1)


    # TESTING PURPOSES
    x_train = x_train.sample(frac=0.01, random_state=42)
    print(x_train.shape)
    x_val = x_val.sample(frac=0.01, random_state=42)
    print(x_val.shape)

    y_train = x_train.copy()
    X_train, _ = SpatialTemporalMask(mask_prob=mask_prob).forward(x_train)

    y_val = x_val.copy()
    X_val, _ = SpatialTemporalMask(mask_prob=mask_prob).forward(x_val)

    model = Transformer(d_model, d_spatial, d_temporal, d_value, ffn_hidden, num_heads, drop_prob, mask_prob, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    log('Training model')
    for epoch in range(25):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            optimizer.zero_grad()
            y_pred = model(X_batch)
            y_batch = torch.tensor(y_batch.values).to(get_device())
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        log(f'Epoch {epoch+1} Loss {loss.item()}')

    log('Evaluating model')
    y_pred = model(X_val)
    y_val = torch.tensor(y_val.values).to(get_device())
    val_loss = criterion(y_pred, y_val)
    log(f'Validation Loss {val_loss.item()}')
    log('Saving model')
    torch.save(model.state_dict(), os.path.join(model_folder, 'model.pth'))