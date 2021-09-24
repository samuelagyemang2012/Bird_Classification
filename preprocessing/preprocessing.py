import librosa
import librosa.display
# import matplotlib.pyplot as plt
import pandas as pd
import os

wav_path = "G:/datasets/birdsounds/training_data/audio/wav/"
wav_files = os.listdir(wav_path)

# data, sample_rate = librosa.load(wav_path + wav_files[0])
# print(sample_rate)
# librosa.display.waveplot(data, sr=sample_rate)

image_df = pd.read_csv("../data/training/image/train.csv")
audio_df = pd.read_csv("../data/training/audio/train.csv")



