import torch
import configparser
import numpy as np

from torch.utils.data import DataLoader 
from src.models.lstm import LSTM_model
from src.utils.dataset import SampleDataset


config = configparser.ConfigParser()
config.read('config.ini')
config_use = config['switcher']['config_use']

test_set = SampleDataset(config[config_use]['save_dataset_path'] + 
                          config[config_use]['src_dataset_path'].split('/')[-1][:-4] + 
                          '_test.npy'
                        )
test_loader = DataLoader(
                          dataset = test_set, 
                          batch_size = int(config[config_use]['test_batch_size']), 
                          shuffle = False
                         )

model = LSTM_model(
    num_classes = int(config[config_use]['num_classes']),
    input_size = int(config[config_use]['input_size']),
    hidden_size = int(config[config_use]['hidden_size']),
    num_layers = int(config[config_use]['num_layers']),
    seq_length = int(config[config_use]['seq_length'])
)
model.load_state_dict(torch.load(config[config_use]['save_model_path']))
model.eval()

for index, data in enumerate(test_loader):
    input = data[:, np.newaxis, :-1]
    output = model(input)

