import pandas as pd
import os
import shutil
import time

selected_species_list = ["Fringilla coelebs", "Parus major", "Turdus merula", "Sylvia communis"]
columns = ["Species", "Label"]

# Audio
selected_audio_path = "../data/species.csv"

audio_path = "G:/datasets/birdsounds/audio/"
audio_dest = "G:/datasets/birdsounds/training_data/audio/mp3"
audio_files = os.listdir(audio_path)

# Images
images_path = "G:/datasets/birdsounds/images"
images_dest = "G:/datasets/birdsounds/training_data/images/"
image_directories = os.listdir(images_path)

# For CSV
images_list = []
audio_list = []

start = time.time()
# Audio
df = pd.read_csv(selected_audio_path)
selected_audios = df["Species"].tolist()

for audio in selected_audios:
    file = audio_path + audio + ".mp3"
    try:
        file_ = audio + ".wav"
        x = audio.split("-")
        label = x[0] + " " + x[1]
        audio_list.append([file_, label])
        shutil.copy(file, audio_dest)
        print("Moved: " + file_)
    except:
        print("Not found: " + file)

audio_df = pd.DataFrame(audio_list, columns=columns, index=None)
audio_df.to_csv("../data/training/all_audio_data.csv", index=False)
print("Audio files moved.")
###################################################

# Images

for d in image_directories:
    subfolder = os.listdir(images_path + "/" + d)

    if d == selected_species_list[0]:
        for f in subfolder:
            f_path = images_path + "/" + d + "/" + f
            file_ = f
            label = selected_species_list[0]
            images_list.append([file_, label])
            try:
                shutil.copy(f_path, images_dest)
                print("Moved: " + f_path)
            except:
                print("Not found: " + f_path)

    if d == selected_species_list[1]:
        for f in subfolder:
            f_path = images_path + "/" + d + "/" + f
            file_ = f
            label = selected_species_list[1]
            images_list.append([file_, label])
            try:
                shutil.copy(f_path, images_dest)
                print("Moved: " + f_path)
            except:
                print("Not found: " + f_path)

    if d == selected_species_list[2]:
        for f in subfolder:
            f_path = images_path + "/" + d + "/" + f
            file_ = f
            label = selected_species_list[2]
            images_list.append([file_, label])
            try:
                shutil.copy(f_path, images_dest)
                print("Moved: " + f_path)
            except:
                print("Not found: " + f_path)

    if d == selected_species_list[3]:
        for f in subfolder:
            f_path = images_path + "/" + d + "/" + f
            file_ = f
            label = selected_species_list[3]
            images_list.append([file_, label])
            try:
                shutil.copy(f_path, images_dest)
                print("Moved: " + f_path)
            except:
                print("Not found: " + f_path)

image_df = pd.DataFrame(images_list, columns=columns, index=None)
image_df.to_csv("../data/training/all_image_data.csv", index=False)
print("Image files moved.")

end = time.time()

print("Took: " + str((end - start)) + "s")
print("Done")
