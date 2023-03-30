import numpy as np
import os
import pandas as pd
from IPython.display import display

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'

emotion_list = ['neu' 'fru' 'xxx' 'sur' 'ang' 'hap' 'sad' 'exc' 'oth' 'fea' 'dis']

zone = {'FH' : ['FH1', 'FH2', 'FH3'],
        'CH' : ['CH1', 'CH2', 'CH3'], 
        'RB' : ['RBM0', 'RBM1', 'RBM2', 'RBM3', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4' ],
        'LB' : ['LBM0', 'LBM1', 'LBM2', 'LBM3', 'LBRO1', 'LBRO2', 'LBRO3', 'LBRO4' ], 
        'N'  : ['MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL'],
        'MOU': ['Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8'], 
        'LC' : ['LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8'],
        'RC' : ['RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8'], 
        'HD' : ['LHD', 'RHD'],
        'LD' : ['LLID', 'RLID'] }

points = ['CH1', 'CH2', 'CH3', 'FH1', 'FH2', 'FH3', 'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 'LC8', 'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 'RC8', 'LLID', 'RLID', 'MH', 'MNOSE', 'LNSTRL', 'TNOSE', 'RNSTRL', 'LBM0', 'LBM1', 'LBM2', 'LBM3', 'RBM0', 'RBM1', 'RBM2', 'RBM3', 'LBRO1', 'LBRO2', 'LBRO3', 'LBRO4', 'RBRO1', 'RBRO2', 'RBRO3', 'RBRO4', 'Mou1', 'Mou2', 'Mou3', 'Mou4', 'Mou5', 'Mou6', 'Mou7', 'Mou8', 'LHD', 'RHD']

zones_interet = ['CH', 'FH', 'LB', 'RB', 'RC', 'LC', 'LD', 'HD', 'MOU', 'N', ]

points_interet = [point for region in zones_interet for point in zone[region]]

def boolean_from_zone(zone, points, zones_interet) : 
    
    def boolean_from_points(points, points_interet) : 
        result = []
        for i in range(len(points)) : 
            if points[i] in points_interet : 
                result.append(True) 
            else : 
                result.append(False)
        return result
    
    points_interet = [point for region in zones_interet for point in zone[region]]
    
    return boolean_from_points(points, points_interet)


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



def get_mocap_rot(path, zone_interet):
    
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
    data_list[np.isnan(data_list)] = 0

    bool_points = boolean_from_zone(zone, points, zone_interet)
    data_list = data_list[ : , bool_points , : ]

    return header, xyz, data_list



def frame_to_s(fr):
    return (fr+2)*10/1000

def filter_by_emotions(df, emotions_of_interest):
    return df[df['emotion'].isin(emotions_of_interest)]

def filter_by_session(df, sessions) : 
     return df[df['session'].isin(sessions)]

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

    path = df['MOCAP_rotated_path'][0]
    print(os.path.join(root_path, path))
    print(calcul_proba(['hap','neu']))

        




# print (data_rot, data_rot.shape)