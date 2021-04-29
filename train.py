import torch
import configparser
import numpy as np
import datetime

from torch.utils.data import DataLoader 
from src.models.lstm import LSTM_model
from src.models.gru import GRU_model
from src.utils.dataset import LocalDataset, OnlineDataset, OnlineSentimentDataset
from src.utils.data_transfer import data_transfer
from src.utils.data_clean import data_clean
from src.utils.sentiment_score import sentiment_score


config = configparser.ConfigParser()
config.read('config.ini')
config_use = config['switcher']['config_use']

# Step 1. Preprepare data 
if config[config_use]['use_sentiment_analysis'] == 'True':
    # data_clean(load_path=config[config_use]['clean_dataset_load_path'], 
    #            save_path=config[config_use]['clean_dataset_save_path']
    #           )
    # sentiment_score(load_path=config[config_use]['clean_dataset_save_path'], 
    #            save_path=config[config_use]['sentiment_dataset_save_path']
    #            )
    target_index = -2

# Step 2. Create Dataset
if config[config_use]['use_local_data'] == 'True': 
    data_transfer(config[config_use]['src_dataset_path'], 
                config[config_use]['save_dataset_path'], 
                float(config[config_use]['train_dataset_rate'])
                )
    train_set = LocalDataset(config[config_use]['save_dataset_path'] + 
                            config[config_use]['src_dataset_path'].split('/')[-1][:-4] + 
                            '_train.npy'
                            )
elif config[config_use]['use_sentiment_analysis'] == 'False':
    train_set = OnlineDataset(config[config_use]['company'], 
                              config[config_use]['data_source'], 
                              datetime.datetime(*(int(config[config_use]['train_strat_date'][:4]), int(config[config_use]['train_strat_date'][4:6]), int(config[config_use]['train_strat_date'][6:]))),
                              datetime.datetime(*(int(config[config_use]['train_end_date'][:4]), int(config[config_use]['train_end_date'][4:6]), int(config[config_use]['train_end_date'][6:])))
                             )
else:
    train_set = OnlineSentimentDataset(config[config_use]['company'], 
                                       config[config_use]['data_source'], 
                                       datetime.datetime(*(int(config[config_use]['train_strat_date'][:4]), int(config[config_use]['train_strat_date'][4:6]), int(config[config_use]['train_strat_date'][6:]))),
                                       datetime.datetime(*(int(config[config_use]['train_end_date'][:4]), int(config[config_use]['train_end_date'][4:6]), int(config[config_use]['train_end_date'][6:]))),
                                       config[config_use]['sentiment_dataset_save_path']
                                       )

# Step 3. Create dataloader
train_loader = DataLoader(
                          dataset = train_set, 
                          batch_size=config[config_use]['batch_size'], 
                          shuffle = False
                         )

# Step 4. Init model
if config[config_use]['module'] == 'gru':
    model = GRU_model(
        num_classes = int(config[config_use]['num_classes']),
        input_size = int(config[config_use]['input_size']),
        hidden_size = int(config[config_use]['hidden_size']),
        num_layers = int(config[config_use]['num_layers']),
        seq_length = int(config[config_use]['seq_length'])
    )
else:
    model = LSTM_model(
        num_classes = int(config[config_use]['num_classes']),
        input_size = int(config[config_use]['input_size']),
        hidden_size = int(config[config_use]['hidden_size']),
        num_layers = int(config[config_use]['num_layers']),
        seq_length = int(config[config_use]['seq_length'])
    )

# Step 5. Init utils
loss_list = []
loss_fuction = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = float(config[config_use]['learning_rate']))

# Step 6. Train
for epoch in range(int(config[config_use]['epoches'])):
    for index, data in enumerate(train_loader):
        input = data[:, np.newaxis, :].float()
        target = data[:, target_index].float()
        output = model(input)

        loss = loss_fuction(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss)
    if epoch % 100 == 0:
        print('Epoch: {}, Loss: {:1.4f}'.format(epoch, loss_list[-1]))

# Step 7. Save model
torch.save(model.state_dict(), config[config_use]['save_model_path'])
