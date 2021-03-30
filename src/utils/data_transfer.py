import datetime
import numpy as np
import pandas as pd


def data_transfer(src_path: str, save_path: str, train_data_rate: float):
    file_name = src_path.split('/')[-1][:-4]

    dataframe = pd.read_csv(src_path)
    for i in range(dataframe.shape[0]):
        dataframe.iloc[i, 0] = datetime.datetime.strptime(dataframe.iloc[i, 0], '%Y-%m-%d').toordinal()
    
    nparray = dataframe.to_numpy()
    train_data_index = int(nparray.shape[0] * train_data_rate)
    np.save(save_path + file_name + '_train.npy', nparray[:train_data_index, :])
    np.save(save_path + file_name + '_test.npy', nparray[train_data_index:, :])