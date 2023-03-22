import numpy as np
import os
import pandas as pd
from IPython.display import display

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'
points = ['CH1', 'CH2', 'CH3', 'FH1', 'FH2', 'FH3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8', 'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8', 'LLID', 'RLID', 'MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL', 'LBM0', 'LBM1', 'LBM2', 'LBM3', 'RBM0', 'RBM1', 'RBM2', 'RBM3', 'LBRO1', 'LBRO2', 'LBRO3', 'LBRO4', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4', 'Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8', 'LHD', 'RHD']


def get_mocap_rot(path):
    
    '''get_mocap_rot permet d'extraire 
    les differentes coordonnees des points du 
    visage pour un enregistrement donne. La fonction 
    prend en argument le path relatif a un enregistrement
    et sort en argument : 
    - le nom des points
    - les intitules des coordonnes de ces points
    - un np.array de taille (nb de frames, nb de points, 3)
    avec pour chaque frame, les 3 coordonnees des points
    du visage'''


    real_path = os.path.join(root_path, path)
    f = open(real_path, 'r').read()
    f = np.array(f.split('\n'))
    header = f[0].split(' ')
    xyz = f[1].split(' ')
    f = f[2:]
    
    data_list = []
    
    for data in f:
        data2 = data.split(' ')
        if(len(data2)<2):
            continue
        coordonnees = data2[2:]
        data_list.append(coordonnees)
    data_list = np.array(data_list)
    data_list = np.reshape(data_list, (-1, len(header)-2 ,3))
    data_list = data_list.astype(np.float32)
    return header, xyz, data_list

def frame_to_s(fr):
    return (fr+2)*10/1000

df = pd.read_csv(os.path.join(root_path, 'iemocap.csv'))


for index, row in df.iterrows():

    session = row['session']
    method = row['method']
    gender = row['gender']
    emotion = row['emotion']
    n_annot = row['emotion']
    agreement = row['agreement']
    wav_path = row['wav_path']

    _, file_name = os.path.split(wav_path)

    break


if __name__ == '__main__' : 


    MOCAP_path = df['MOCAP_rotated_path']
    emotions_results = df['emotion']

    print(get_mocap_rot(MOCAP_path[0])[0])


        




# print (data_rot, data_rot.shape)