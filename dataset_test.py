import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import pandas as pd
import base
from torch.utils.data import DataLoader
import numpy as np

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv')

class IEMOCAP_dataset(Dataset):

    def __init__(self, info_path, training):
        self.info_path = info_path
        self.training = training

        self.dataframe = pd.read_csv(self.info_path)  # this is the df from base

        if self.training :
            self.data_paths = self.dataframe['MOCAP_rotated_path'][:7870]
            self.labels = self.dataframe['emotion'][:7870]
        else:
            self.data_paths = self.dataframe['MOCAP_rotated_path'][7870:].reset_index(drop = True)
            self.labels = self.dataframe['emotion'][7870:].reset_index(drop = True)


    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx): 
        item = torch.from_numpy(base.get_mocap_rot(self.data_paths[idx])[2])
        label = self.labels[idx]
        
        return item, label

iemocap_dataset = IEMOCAP_dataset(info_path, True)

training_data = IEMOCAP_dataset(info_path, True)
test_data = IEMOCAP_dataset(info_path, False)


         
a = training_data[0][0]
b = training_data[1][0]

print("size(a) is {} and size(b) is {}".format(a.shape, b.shape))

p = pad_sequence([a,b], batch_first=True)

print("padded shape is ", p.shape)


def collate_fn(batch) :
    sequence = [batch[i][0] for i in range (len(batch))]
    padded = pad_sequence(sequence, batch_first=True)
    return padded


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False,collate_fn=collate_fn)

a = iter(train_dataloader)

print(next(a).shape, next(a).shape)



