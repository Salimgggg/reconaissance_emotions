import numpy as np
import os
import pandas
from IPython.display import display

def get_labels(annot_file, file_name):
    
    f = open(annot_file, 'r').read()
    f = f.split('\n')
    f = f[2:]
    
    for data in f:
        
        if len(data) > 0:
            if data[0] == '[':
                data2 = data.split('\t')
                
                if data2[1] == file_name:
                    emo = data2[2]
                    vad = data2[3][1:-1].split(', ')
                    return emo, [float(x) for x in vad]
        
    raise ValueError('Label not found')

def get_mocap_rot(path):

    f = open(path, 'r').read()
    f = np.array(f.split('\n'))
    header = f[0].split(' ')
    xyz = f[1].split(' ')
    f = f[2:]
    
    data_list = []
    
    for data in f:
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        dic = {'frame': data2[0], 'time': data2[1], 
               'markers': np.array(data2[2:]).astype(float)}
        data_list.append(dic)
        
    return header, xyz, data_list

def get_ph_fa(path):
    f = open(path, 'r').read()
    f = np.array(f.split('\n'))
    header = f[0].split()
    f = f[1:-2]
    data_list = []
    
    for data in f:
        data2 = data.split()
        dic = {'SFrm':np.array(data2[0]).astype(int), 
               'EFrm':np.array(data2[1]).astype(int), 
               'SegAScr':np.array(data2[2]).astype(int), 
               'Phone':data2[3]}
        data_list.append(dic)
    
    return header, data_list

def frame_to_s(fr):
    return (fr+2)*10/1000


root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'

df = pandas.read_csv(os.path.join(root_path, 'iemocap.csv'))

display(df)