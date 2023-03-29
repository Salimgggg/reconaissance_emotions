import pandas as pd
import os 
import base

root_path = 'IEMOCAP_full_release_withoutVideos_sentenceOnly'


df = pd.read_csv(os.path.join(root_path, 'iemocap.csv'))

mocap_row = 'MOCAP_rotated_path'
wav_row = 'wav_path'
filter_session = base.filter_by_session


filtered_df = df[df['emotion'] == 'sad']
test_session = filter_session(df, [5])
df_1 = test_session[test_session['emotion'] == 'exc']

print(len(df_1))
print((245)/(245+299)) 






# print(df[mocap_row])
# diff_count = df[mocap_row].ne(df[wav_row])
# print(diff_count[diff_count == True])

