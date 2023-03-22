import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import base
from torch.utils.data import DataLoader
import numpy as np




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
# pour l'instant on normalise (x-mean)/std sur la frame entière

    def __getitem__(self, idx): 
        item = base.get_mocap_rot(self.data_paths[idx])[2]
        label = self.labels[idx]
        
        data_mean = np.mean(item, 0)
        data_std = np.std(item, 0)
        normalized_data = (item - data_mean)/data_std
        item_tensor = torch.Tensor(normalized_data)
        return item_tensor, label
   


root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv')

iemocap_dataset = IEMOCAP_dataset(info_path, True)

print(len(iemocap_dataset))
print(iemocap_dataset[0])

training_data = IEMOCAP_dataset(info_path, True)
test_data = IEMOCAP_dataset(info_path, False)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

print(next(iter(test_dataloader)))