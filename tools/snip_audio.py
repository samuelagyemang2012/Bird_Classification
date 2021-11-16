import os

audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/audio/wav_files/"
audio_dest = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/audio/clipped_wav/"

data = os.listdir(audio_source)

for d in data:
    ext = d.split(".")

    if ext[1] == "wav":
        input_path = audio_source + d
        output_path = audio_dest + d
    else:
        input_path = audio_source + d
        output_path = audio_dest + ext[0] + ".wav"

    cmd = "ffmpeg -i " + input_path + " -ss 00:00:00 -t 00:00:01 " + output_path
    os.system(cmd)

print("done")
