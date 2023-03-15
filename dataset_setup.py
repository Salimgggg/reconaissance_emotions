import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import base
import os
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
    
    def __getitem__(self, idx) : 
        item = base.get_mocap_rot(self.data_paths[idx])[2]
        label = self.labels[idx]
        return torch.from_numpy(item), label
    



root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv')

iemocap_dataset = IEMOCAP_dataset(info_path)

print(iemocap_dataset.__len__())

example = iemocap_dataset.__getitem__(4)

print(example)











