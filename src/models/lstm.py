import torch
import torch.nn as nn

from torch.autograd import Variable 


class LSTM_model(nn.Module):
  def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
    super(LSTM_model, self).__init__()
    self.num_classes = num_classes
    self.num_layers = num_layers
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.seq_length = seq_length
 
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes) 

  def forward(self,x):
    h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
    c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
   
    x, _ = self.lstm(x, (h0, c0))
    x = self.fc(x[:, -1, :])
    
    return x