import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import base
from torch.utils.data import DataLoader

#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)



class IEMOCAP_dataset(Dataset) : 

    def __init__(self, info_path,test) :
        self.info = pd.read_csv(info_path) # this is the df from base
        if test == True :
            self.data_paths = self.info['MOCAP_rotated_path'][:7870]
        else : 
            self.data_paths = self.info['MOCAP_rotated_path'][7870:]       
        self.labels = self.info['emotion']

    def __len__(self) : 
        return len(self.labels)
# pour l'instant on normalise (x-mean)/std sur la frame entière
    def __getitem__(self, idx) : 
        item = base.get_mocap_rot(self.data_paths[idx])[2]
        label = self.labels[idx]
        item_tensor = torch.Tensor(item)
        data_mean = torch.mean(item_tensor)
        data_std = torch.std(item_tensor)
        normalized_data = (item_tensor - data_mean) / data_std
        return normalized_data, label
    



root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv')

iemocap_dataset = IEMOCAP_dataset(info_path,True)

print(iemocap_dataset.__len__())

example = iemocap_dataset.__getitem__(4)

print(example)











