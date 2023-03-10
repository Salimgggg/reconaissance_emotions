import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
import base
import os



class Dataset : 

    def __init__(self, info_path) :
        self.info = pd.read_csv(info_path) # this is the df from base
        self.data_paths = self.info['MOCAP_rotated_path']
        self.labels = self.info['emotion']

    def __len__(self) : 
        return len(self.labels)
    
    def __getitem__(self, idx) : 
        item = base.get_mocap_rot(self.data_paths[idx])[2]
        label = self.labels[idx]
        return torch.from_numpy(item), label

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv')

iemocap_dataset = Dataset(info_path)

print(iemocap_dataset.__len__())

example = iemocap_dataset.__getitem__(4)[0].shape

print(example)











