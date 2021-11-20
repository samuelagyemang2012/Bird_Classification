import librosa.display
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os

hop_length_ = 512
n_fft_ = 2048
n_mels_ = 128
n_mfcc = 40

audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/audio/clipped_wav/"
audio_dest = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/audio/spectrograms/"
MODE = "MFCC"  # MFCC

MFCC = []


def generate_mel_spectrogram(wav_file, n_fft, hop_length, n_mels, path):
    samples, sample_rate = librosa.load(wav_file)
    data, _ = librosa.effects.trim(samples)

    S = librosa.feature.melspectrogram(data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length)  # , x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)


def generate_mfcc(wav_file, n_mfcc):
    audio_data, sampling_rate = librosa.load(wav_file)
    mfcc_ = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=n_mfcc)
    scaled_mfcc = np.mean(mfcc_.T, axis=0)
    return scaled_mfcc


if MODE == "SPEC":
    wav_files = os.listdir(audio_source)
    for w in wav_files:
        path = audio_dest + w.split(".")[0] + ".png"
        generate_mel_spectrogram(audio_source + w, n_fft_, hop_length_, n_mels_, path)

if MODE == "MFCC":
    bird_audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/audio/bird/clipped_wav/"
    dog_audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/audio/dog/clipped_wav/"
    car_audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/audio/car/clipped_wav/"

    bird_files = os.listdir(bird_audio_source)
    dog_files = os.listdir(dog_audio_source)
    car_files = os.listdir(car_audio_source)
    MFCC = []

    for b in bird_files:
        mfcc = generate_mfcc(bird_audio_source + b, n_mfcc)
        MFCC.append([b, np.array(mfcc).astype(np.float32), "bird"])
    print("bird mfcc complete")

    for d in dog_files:
        mfcc = generate_mfcc(dog_audio_source + d, n_mfcc)
        MFCC.append([d, np.array(mfcc).astype(np.float32), "dog"])
    print("dog mfcc complete")

    for c in car_files:
        mfcc = generate_mfcc(car_audio_source + c, n_mfcc)
        MFCC.append([c, np.array(mfcc).astype(np.float32), "car"])
    print("car mfcc complete")

all_df = pd.DataFrame(MFCC, index=None, columns=["file", "mfcc", "class"])
all_df.to_csv("../data/all_mfcc.csv", index=False)
