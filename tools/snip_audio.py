import os

audio_source = "E:/datasets/birdsounds/training_data/audio/wav/"
audio_dest = "E:/datasets/birdsounds/training_data/audio/trimmed_wav/"

data = os.listdir(audio_source)

for d in data:
    cmd = "ffmpeg -i " + audio_source + d + " -ss 00:00:00 -t 00:00:05 " + audio_dest + d
    os.system(cmd)

print("done")
