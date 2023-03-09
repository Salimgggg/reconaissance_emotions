import numpy as np
import os
import pandas as pd
from IPython.display import display


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
        coordonnees = data2[2:]
        data_list.append(coordonnees)
    data_list = np.array(data_list)
    data_list = np.reshape(data_list, (-1, len(header)-2 ,3))


        
    return header, xyz, data_list


def frame_to_s(fr):
    return (fr+2)*10/1000


root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'

df = pd.read_csv(os.path.join(root_path, 'iemocap.csv'))

# display(df)





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

MOCAP_path = df['MOCAP_rotated_path']
emotions_results = df['emotion']

#display the number of each emotion in the dataset  


print(emotions_results.value_counts())

def global_mocap_info (list_paths) : 
    for path in list_paths : 
        header, xyz, data = get_mocap_rot(path)
        
        



# print (data_rot, data_rot.shape)