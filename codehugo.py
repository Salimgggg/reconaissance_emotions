import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import math
from base import get_mocap_rot
import torch.nn as nn
#train_data = get_mocap_rot(
    #'/workspaces/reconaissance_emotions/IEMOCAP_full_release_withoutVideos_sentenceOnly/IEMOCAP_full_release/Session1/sentences/MOCAP_rotated/Ses01F_impro01/Ses01F_impro01_F000.txt')[2]


#sample = 2507*['xxx']+1849*['fru']+1708*['neu']+1103*['ang']+1084*['sad']+1041*['exc']+595*['hap']+107*['sur']+40*['fea']+3*['oth']+2*['dis']


def calcul_proba(liste_emotions):
    proba_emotion={
        'fru' : 1849,
        'neu' : 1708,
        'ang' : 1103,
        'sad' : 1084,
        'exc' : 1041,
        'hap' : 595,
        'sur' : 107,
        'fea' : 40,
        'oth' : 3,
        'dis' : 2
    }
    tot=0
    sample=[]
    for k in range(len(liste_emotions)):
        emotion=liste_emotions[k]
        sample += proba_emotion[emotion]*[emotion]
        tot+=proba_emotion[emotion]
    for i in range(10000):
        liste=[]
        occ=0
        rand=np.random.choice(liste_emotions, tot, replace = True)
        for j in range(tot):
            if sample[j]==rand[j]:
                occ+=1
        pourcentage=occ/tot
        liste.append(pourcentage)
    proba=np.mean(liste)
    return proba 
print(calcul_proba(['hap','neu']))