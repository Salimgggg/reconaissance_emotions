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
        '''training is a boolean, specifying if we want the training or test dataset
        The first 7869 datapoints are from the 4 first sessions, and the last ones are the
        5th session. We want to train on the first 4, and test on the 5th.'''
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

'''We instantiate the training and test datasets using the info_path '''

training_data = IEMOCAP_dataset(info_path, True)
test_data = IEMOCAP_dataset(info_path, False)

'''We are feeding different size datapoints to the dataloader which doesn't support that with
the default collate_fn function, so we define a custom collate_fn to zero_pad every batch following
the datapoint with the most frames of this batch'''

def collate_fn(batch) :
    sequence = [batch[i][0] for i in range (len(batch))]
    padded = pad_sequence(sequence, batch_first=True)
    return padded


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False,collate_fn=collate_fn)





