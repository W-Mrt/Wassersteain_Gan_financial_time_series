import torch
import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm
from torch.nn import LSTM, Module, MaxPool2d, Sequential, Conv2d, BatchNorm2d, ReLU 


###---CNN____model1
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.main = nn.Sequential(nn.Linear(50, 100),
#                          nn.LeakyReLU(0.2, inplace=True),
#                          AddDimension(),
#                          spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
#                          nn.Upsample(200),

#                          spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
#                          nn.LeakyReLU(0.2, inplace=True),
#                          nn.Upsample(400),

#                          spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
#                          nn.LeakyReLU(0.2, inplace=True),
#                          nn.Upsample(800),

#                          spectral_norm(nn.Conv1d(32, 1, 3, padding=1), n_power_iterations=10),
#                          nn.LeakyReLU(0.2, inplace=True),

#                          SqueezeDimension(),
#                          nn.Linear(800, 100)
#                          )

#     def forward(self, input):
#         return self.main(input)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.main = nn.Sequential(AddDimension(),
#                          spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
#                          nn.LeakyReLU(0.2, inplace=True),
#                          nn.MaxPool1d(2),

#                          spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
#                          nn.LeakyReLU(0.2, inplace=True),
#                          nn.MaxPool1d(2),

#                          spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
#                          nn.LeakyReLU(0.2, inplace=True),
#                          nn.Flatten(),

#                          nn.Linear(800, 50),
#                          nn.LeakyReLU(0.2, inplace=True),

#                          nn.Linear(50, 15),
#                          nn.LeakyReLU(0.2, inplace=True),

#                          nn.Linear(15, 1)
#                          )

#     def forward(self, input):
#         return self.main(input)


##----MLP________model2
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(50, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 400),
            nn.LeakyReLU(0.2, inplace=True),    
            nn.Linear(400, 600),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(600, 800),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(800, 100),
           
        )

    def forward(self, input):
        return self.main(input)

        

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.main = nn.Sequential(
            AddDimension(),
            spectral_norm(nn.Conv1d(1, 32, 3, padding=1), n_power_iterations=10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),

            spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),

            spectral_norm(nn.Conv1d(32, 32, 3, padding=1), n_power_iterations=10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(800, 400),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(400, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 15),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(15, 1)
          

        )

    def forward(self, input):
        return self.main(input)

# # # ####________model3____LSTM
# class Generator(nn.Module):
#     """sequence of noise vectors as input
#     self.args:
#         in_dim: Input noise dim
#         out_dim: Output dim
#         n_layers: number of lstm layers
#         hidden_dim: dim of lstm hidden layer
#     Input: shape (batch_size, seq_len, in_dim)
#     Output: shape (batch_size, seq_len, out_dim)
#     """

#     def __init__(self):
#         super().__init__()
#         self.in_dim = 1
#         self.n_layers = 4
#         self.hidden_dim = 32
#         self.out_dim = 1

#         self.lstm = nn.LSTM(self.in_dim, self.hidden_dim, self.n_layers, batch_first=True)
#         self.linear = nn.Sequential(nn.Linear(self.hidden_dim, self.out_dim), nn.Tanh())

#     def forward(self, input):
#         batch_size, seq_len = input.size(0), input.size(1)
#         h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
#         c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
#         recurrent_features, _ = self.lstm(input, (h_0, c_0))
#         outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
#         outputs = outputs.view(batch_size, seq_len, self.out_dim)
#         return outputs


# class Discriminator(nn.Module):
#     """sequence as input, outputs a probability for each element
#     self.args:
#         in_dim: input noise dim
#         n_layers: number lstm layers
#         hidden_dim: dim of lstm hidden layer 
#     Inputs: shape (batch_size, seq_len, in_dim)
#     Output: shape (batch_size, seq_len, 1)
#     """

#     def __init__(self):
#         super().__init__()
#         self.in_dim = 1
#         self.n_layers = 4
#         self.hidden_dim = 32

#         self.lstm = nn.LSTM(self.in_dim, self.hidden_dim, self.n_layers, batch_first=True)
#         self.linear = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())

#     def forward(self, input):
#         batch_size, seq_len = input.size(0), input.size(1)
#         h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
#         c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

#         recurrent_features, _ = self.lstm(input, (h_0, c_0))
#         outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
#         outputs = outputs.view(batch_size, seq_len, 1)
#         return outputs



class AddDimension(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class SqueezeDimension(nn.Module):
    def forward(self, x):
        return x.squeeze(1)
