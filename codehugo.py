import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import math
from base import get_mocap_rot
import torch.nn as nn
#train_data = get_mocap_rot(
    '/workspaces/reconaissance_emotions/IEMOCAP_full_release_withoutVideos_sentenceOnly/IEMOCAP_full_release/Session1/sentences/MOCAP_rotated/Ses01F_impro01/Ses01F_impro01_F000.txt')[2]

sample = 2507*['xxx']+1849*['fru']+1708*['neu']+1103*['ang']+1084*['sad']+1041*['exc']+595*['hap']+107*['sur']+40*['fea']+3*['oth']+2*['dis']
#sample = 2507*['xxx']+1849*['fru']+1708*['neu']+1103*['ang']+1084*['sad']+1041*['exc']+595*['hap']+107*['sur']+40*['fea']+3*['oth']+2*['dis']
tot=len(sample)
liste=[]
for i in range(1000):
    rand=np.random.choice(['xxx', 'fru', 'neu', 'ang','sad','exc','hap','sur','fea','oth','dis'], tot, replace = True, p = [2507/tot, 1849/tot, 1708/tot,1103/tot,1084/tot,1041/tot,595/tot,107/tot,40/tot,3/tot,2/tot])
    occ=0
    for j in range(tot):
        if sample[j]==rand[j]:
            occ+=1
    pourcentage=occ/tot
    liste.append(pourcentage)
proba=np.mean(liste)
print(proba)