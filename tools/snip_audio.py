import os
import pandas as pd

audio_source = "G:/datasets/birdsounds/audio/"
data_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/multimodal_project/data/training/all_audio_data.csv"
train_data = pd.read_csv(data_path)["Species"].tolist()
audio_dest = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/training_data/new_audio/"

data = os.listdir(audio_source)
print(len(train_data))
for d in train_data:
    file = d.split(".")[0]
    input_path = audio_source + file + ".mp3"
    output_path = audio_dest + file + ".wav"
    print(output_path)
    cmd = "ffmpeg -i " + input_path + " -ss 00:00:00 -t 00:00:10 " + output_path
    os.system(cmd)

print("done")
