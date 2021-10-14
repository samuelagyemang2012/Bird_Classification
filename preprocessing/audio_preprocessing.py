import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

hop_length_ = 512
n_fft_ = 2048
n_mels_ = 128

audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/training_data/new_audio/"
audio_dest = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/training_data/new_spectograms/"


def generate_mel_spectrogram(wav_file, n_fft, hop_length, n_mels, path):
    samples, sample_rate = librosa.load(wav_file)
    data, _ = librosa.effects.trim(samples)

    S = librosa.feature.melspectrogram(data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length)  # , x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+2.0f dB')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)


wav_files = os.listdir(audio_source)

for w in wav_files:
    path = audio_dest + w.split(".")[0] + ".png"
    generate_mel_spectrogram(audio_source + w, n_fft_, hop_length_, n_mels_, path)

print("done")
