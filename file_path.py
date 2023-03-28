import pandas as pd
import os 

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'


df = pd.read_csv(os.path.join(root_path, 'iemocap.csv'))

mocap_row = 'MOCAP_rotated_path'
wav_row = 'wav_path'

filtered_df = df[df[mocap_row] != 'missing']





# print(df[mocap_row])
# diff_count = df[mocap_row].ne(df[wav_row])
# print(diff_count[diff_count == True])

