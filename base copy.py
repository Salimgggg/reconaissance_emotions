import numpy as np
import os
import pandas as pd
from IPython.display import display

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'


def generate_tensor(full_path, points_list):
    f = open(full_path, "r")
    data = f.readlines()
    data = [i.replace('\n', '') for i in data]
    for i in range(len(data)):
        data[i] = data[i].split(' ')
    #nb de frames, points, 
    df = pd.DataFrame(index = ['x','y','z'], columns = data[0][2:])
    video = {}
    for frame in data[2:]:
        points = [frame[i:i+3] for i in range(2,len(frame),3)]
        #points = slice_per(frame[2:], int(len(frame[2:])/3))
        point = 0
        for column in df.columns:
            df[column] = points[point]
            point = point + 1
        video[frame[0]] = {'time': frame[1], 'points': df.astype(float)}


    tensor = np.zeros((0,len(points_list),3))
    for frame in video:
        mat = video[frame]['points'][points_list].to_numpy().T
        tensor = np.concatenate((tensor, np.expand_dims(mat, axis=0)), axis=0)
    return tensor



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

    print(generate_tensor(MOCAP_path[0]))


        




# print (data_rot, data_rot.shape)