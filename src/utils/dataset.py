import numpy as np 

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler 


class SampleDataset(Dataset):
    def __init__(self, path: str) -> None:
        min_max_scaler = MinMaxScaler()
        self.data = min_max_scaler.fit_transform(np.load(path, allow_pickle=True))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]