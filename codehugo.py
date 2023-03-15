import torch
import torchvision
import torchvision.transforms as transforms
import math
from base import get_mocap_rot
import torch.nn as nn
train_data = get_mocap_rot(
    '/workspaces/reconaissance_emotions/IEMOCAP_full_release_withoutVideos_sentenceOnly/IEMOCAP_full_release/Session1/sentences/MOCAP_rotated/Ses01F_impro01/Ses01F_impro01_F000.txt')[2]
train_data_tensor = torch.Tensor(train_data)
data_mean = torch.mean(train_data_tensor)
data_std = torch.std(train_data_tensor)
normalized_data = (train_data_tensor - data_mean) / data_std
print(normalized_data)
# print(train_data)
# val_data_tensor = torch.Tensor(val_data)
# test_data_tensor = torch.Tensor(test_data)

# train_labels_tensor = torch.Tensor(train_labels).long()
# val_labels_tensor = torch.Tensor(val_labels).long()
# test_labels_tensor = torch.Tensor(test_labels).long()
"""""
def preprocess_motion_capture_data(data):
    # Convertir les données en tensor PyTorch
    data_tensor = torch.from_numpy(data)

    # Soustraire la moyenne et diviser par l'écart type de chaque dimension
    data_mean = torch.mean(data_tensor, dim=0)
    data_std = torch.std(data_tensor, dim=0)
    normalized_data = (data_tensor - data_mean) / data_std

    # Ajouter une dimension supplémentaire pour représenter le batch
    normalized_data = normalized_data.unsqueeze(0)

    return normalized_data
h,xyz,datas=get_mocap_rot('IEMOCAP_full_release_withoutVideos_sentenceOnly/IEMOCAP_full_release/Session1/sentences/MOCAP_rotated/Ses01F_impro01/Ses01F_impro01_F000.txt')
"""
# print(preprocess_motion_capture_data(datas[:]))
"""""

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)
"""
