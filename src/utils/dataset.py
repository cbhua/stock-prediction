import numpy as np 
import pandas as pd
import pandas_datareader.data as pdr

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class LocalDataset(Dataset):
    def __init__(self, path: str) -> None:
        min_max_scaler = MinMaxScaler()
        self.data = min_max_scaler.fit_transform(np.load(path, allow_pickle=True))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class OnlineDataset(Dataset):
    def __init__(self, company, source, start_date, end_date) -> None:
        min_max_scaler = MinMaxScaler()
        df = pdr.DataReader(company, source, start_date, end_date)
        self.feature = min_max_scaler.fit_transform(df)
        self.target = min_max_scaler.fit_transform(df.iloc[:, 5:6])
        self.data = np.hstack((self.feature[:, :-1], self.target[:]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


class OnlineSentimentDataset(Dataset):
    def __init__(self, company, source, start_date, end_date, sentiment_score_path) -> None:
        min_max_scaler = MinMaxScaler()
        df = pdr.DataReader(company, source, start_date, end_date)

        sentiment_score = pd.read_csv(sentiment_score_path, index_col=0)
        sentiment_score = sentiment_score.drop(columns={'sentiment'})
        sentiment_score = sentiment_score.reset_index(['date'])
        sentiment_score['date'] = pd.to_datetime(sentiment_score['date'])
        sentiment_score = sentiment_score.rename(columns={'date': 'Date', 'sentiment_final': 'Sentiment Final'})
        sentiment_score = sentiment_score.set_index(['Date'])

        self.data = pd.merge(df, sentiment_score, on='Date').to_numpy()


        print('stop')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
