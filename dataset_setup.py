import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import pandas as pd
from base import get_mocap_rot
import os



class Dataset : 

    def __init__(self, root_path, list_of_paths) :
        self.root_path = root_path
        self.list_of_paths = list_of_paths
        self.info = pd.read_csv(os.path.join(root_path, self.list_of_paths))

    def __len__(self) : 
        return len(self.label)
    
    def __getitem__(self, id) : 
        item_path = self.data_paths[id]
        item = get_mocap_rot()
        return item



iemocap_dataset = Dataset('IEMOCAP_full_release_withoutVideos_sentenceOnly', 'iemocap.csv')













