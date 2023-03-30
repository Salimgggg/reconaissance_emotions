import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import pandas as pd
import base
from base import filter_by_emotions, filter_by_session, get_mocap_rot
from torch.utils.data import DataLoader
import numpy as np

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv') 
emotion_list = ['neu', 'fru', 'xxx', 'sur', 'ang', 'hap', 'sad', 'exc', 'oth', 'fea', 'dis'] 
emotions_of_interest = ['sad', 'exc'] 
label_map = { x : (emotions_of_interest.index(x)) for x in emotions_of_interest} 
input_size = len(base.points_interet)*3
window_size = 300

def normalize_array(arr):
    max_vals = np.amax(arr, axis=(0,1)) # maximum values for each coordinate across all frames and points
    normalized_arr = arr / max_vals # divide every coordinate by the maximum value
    
    return normalized_arr


class IEMOCAP_dataset(Dataset):
    
    def __init__(self, info_path, training, emotions_of_interest):
        '''training is a boolean, specifying if we want the training or test dataset
        The first 7869 datapoints are from the 4 first sessions, and the last ones are the
        5th session. We want to train on the first 4, and test on the 5th.'''
        self.emotions_of_interest = emotions_of_interest
        self.info_path = info_path
        self.training = training
        self.dataframe = pd.read_csv(self.info_path) # this is the df from base
        self.dataframe = self.dataframe[self.dataframe['MOCAP_rotated_path'] != 'missing']  
        self.dataframe = self.dataframe[['session', 'MOCAP_rotated_path', 'emotion']]

        if self.training :
            self.sessions = [1, 2, 3, 4]
        else:
            self.sessions = [5]
        
        self.dataframe = filter_by_session(self.dataframe, self.sessions)
        self.dataframe = filter_by_emotions(self.dataframe, self.emotions_of_interest)


        self.data_paths = self.dataframe['MOCAP_rotated_path'].reset_index(drop = True)
        self.labels = self.dataframe['emotion'].reset_index(drop = True)


    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx): 
        
        item = get_mocap_rot(self.data_paths[idx], base.zones_interet)[2]
        item = normalize_array(item)
        item = torch.from_numpy(item)
        label = self.labels[idx]
        
        return item, label

'''We instantiate the training and test datasets using the info_path '''

training_data = IEMOCAP_dataset(info_path, True, emotions_of_interest)
test_data = IEMOCAP_dataset(info_path, False, emotions_of_interest)

'''We are feeding different size datapoints to the dataloader which doesn't support that with
the default collate_fn function, so we define a custom collate_fn to zero_pad every batch following
the datapoint with the most frames of this batch'''

def collate_fn1(batch):
    labels = []
    sequence = []
    for i in range(len(batch)):
        element = batch[i][0]
        label = batch[i][1]
        size = element.shape
        if size[0] > window_size:
            new = element[0:window_size, :, :]
        else:
            new = np.zeros((window_size, size[1], size[2]))
            new[:element.shape[0], :, :] = element
        new = new.reshape(new.shape[0], -1)
        sequence.append(np.array(new))
        labels.append(label_map[label])

    sequence = np.array(sequence)
    sequence = torch.tensor(sequence).float()  # Change data type to torch.FloatTensor
    labels = np.array(labels)
    labels = torch.tensor(labels).to(torch.long)

    return sequence, labels

def collate_fn2(batch): 
    labels = []
    sequence = []
    for i in range(len(batch)):
        element = batch[i][0]
        label = batch[i][1]
        size = element.shape
        if size[0] > window_size:
            min = size[0]//2 - window_size//2
            max = size[0]//2 + window_size//2
            new = element[min:max, :, :]
        else:
            new = np.zeros((window_size, size[1], size[2]))
            new[:element.shape[0], :, :] = element
        new = new.reshape(new.shape[0], -1)
        sequence.append(np.array(new))
        labels.append(label_map[label])

    sequence = np.array(sequence)
    sequence = torch.tensor(sequence).float()  # Change data type to torch.FloatTensor
    labels = np.array(labels)
    labels = torch.tensor(labels).to(torch.long)

    return sequence, labels


training_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, collate_fn=collate_fn2)
test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False,collate_fn=collate_fn2)



if __name__ == '__main__' : 
    a = next(iter(training_dataloader))
    print(a[0].shape)
    print(label_map)


    # Iterate over the dataloader and inspect each batch
    pass
    

    # for data_point in training_data : 
    #     print(data_point[1])
    #     if data_point[1] == 'hap' : 
    #         happy += 1
    #     else :
    #         neutre += 1
    #     print(f'happy : {happy}', f'neutre : {neutre}')
    
         
    # for i, (data, labels) in enumerate(training_dataloader):
    #     print(data[:,:,:,:].shape)




