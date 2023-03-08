import torch
import torchvision
import torchvision.transforms as transforms
import math
from base import *


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

print(preprocess_motion_capture_data(datas[:]))