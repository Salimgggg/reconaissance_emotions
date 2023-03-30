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
from dataset_setup import IEMOCAP_dataset


root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
info_path = os.path.join(root_path, 'iemocap.csv') 

emotion_list = ['neu', 'fru', 'xxx', 'sur', 'ang', 'hap', 'sad', 'exc', 'oth', 'fea', 'dis'] 
emotions_of_interest = ['sad', 'exc', 'neu', 'ang'] 
window_size = 300

emotion = 'emotion'
mocap = 'MOCAP_rotated_path'

df = pd.read_csv(info_path)


training_data = IEMOCAP_dataset(info_path, True, emotions_of_interest)
test_data = IEMOCAP_dataset(info_path, False, emotions_of_interest)


if __name__ == '__main__' : 
    # training_session = filter_by_session(df, [1, 2, 3, 4]).reset_index(drop=True)
    # training_session = filter_by_emotions(training_session, emotions_of_interest)
    # test_session = filter_by_session(df, [5]).reset_index(drop=True)
    # test_session = filter_by_emotions(test_session, emotions_of_interest)
    session_lenghts = []
    for i in range (len(training_data)) : 
        data = training_data[i][0]
        lenght = data.shape[0]
        print(lenght)
        session_lenghts.append(lenght)
    session_lenghts.sort()
    print(session_lenghts)
    bins = np.arange(0, max(session_lenghts) + 30, 30)  # create bins with width of 20
    hist, edges = np.histogram(session_lenghts, bins)  # count number of values in each bin

    plt.bar(range(len(hist)), hist)  # plot bar chart
    plt.xticks(range(len(hist)), edges[:-1])  # label x-axis with bin edges
    plt.show()

  


    pass



