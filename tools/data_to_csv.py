import pandas as pd
import os
import shutil
import time

classes = ["bird", "dog"]
DEST_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/data_csv/"

# Audio
audio_bird_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/audio/spectrograms/"
audio_dog_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dog/audio/spectrograms/"

# Images
image_bird_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/images/"
image_dog_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dog/images/"

# Annotations
bird_annotations_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/annotations/"
dog_annotations_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dog/annotations/"

# Audio data
audio_bird_data = os.listdir(audio_bird_path)
audio_dog_data = os.listdir(audio_dog_path)

# Image data
image_bird_data = os.listdir(image_bird_path)
image_dog_data = os.listdir(image_dog_path)

# Image annotations
bird_ann = os.listdir(bird_annotations_path)
dog_ann = os.listdir(dog_annotations_path)

# For CSV
audio_list = audio_bird_data + audio_dog_data
audio_labels = ["bird" for i in range(len(audio_bird_data))] + ["dog" for j in range(len(audio_dog_data))]

images_list = image_bird_data + image_dog_data[0:945]
images_labels = ["bird" for a in range(len(image_bird_data))] + ["dog" for b in range(len(image_dog_data[0:945]))]

annotations_list = bird_ann + dog_ann[0:945]

print("audio_data", len(audio_list))
print("audio_labels", len(audio_labels))
print("image_data", len(images_list))
print("image_labels", len(images_labels))
print("annotations", len(annotations_list))

all_audio = []
all_images = []

for i, a in enumerate(audio_list):
    all_audio.append([a, audio_labels[i]])

for j, i in enumerate(images_list):
    all_images.append([i, images_labels[j], annotations_list[j]])

all_audio_df = pd.DataFrame(all_audio, columns=['file', 'class'], index=None)
all_audio_df.to_csv("../data/all_audio_data.csv", index=False)

all_image_df = pd.DataFrame(all_images, columns=['file', 'class', 'bbox'], index=None)
all_image_df.to_csv("../data/all_image_data.csv", index=False)

print("Done")
# start = time.time()
# # Audio
# df = pd.read_csv(selected_audio_path)
# selected_audios = df["Species"].tolist()
#
# for audio in selected_audios:
#     file = audio_path + audio + ".mp3"
#     try:
#         file_ = audio + ".wav"
#         x = audio.split("-")
#         label = x[0] + " " + x[1]
#         audio_list.append([file_, label])
#         shutil.copy(file, audio_dest)
#         print("Moved: " + file_)
#     except:
#         print("Not found: " + file)
#
# audio_df = pd.DataFrame(audio_list, columns=columns, index=None)
# audio_df.to_csv("../data/training/all_audio_data.csv", index=False)
# print("Audio files moved.")
# ###################################################
#
# # Images
#
# for d in image_directories:
#     subfolder = os.listdir(images_path + "/" + d)
#
#     if d == selected_species_list[0]:
#         for f in subfolder:
#             f_path = images_path + "/" + d + "/" + f
#             file_ = f
#             label = selected_species_list[0]
#             images_list.append([file_, label])
#             try:
#                 shutil.copy(f_path, images_dest)
#                 print("Moved: " + f_path)
#             except:
#                 print("Not found: " + f_path)
#
#     if d == selected_species_list[1]:
#         for f in subfolder:
#             f_path = images_path + "/" + d + "/" + f
#             file_ = f
#             label = selected_species_list[1]
#             images_list.append([file_, label])
#             try:
#                 shutil.copy(f_path, images_dest)
#                 print("Moved: " + f_path)
#             except:
#                 print("Not found: " + f_path)
#
#     if d == selected_species_list[2]:
#         for f in subfolder:
#             f_path = images_path + "/" + d + "/" + f
#             file_ = f
#             label = selected_species_list[2]
#             images_list.append([file_, label])
#             try:
#                 shutil.copy(f_path, images_dest)
#                 print("Moved: " + f_path)
#             except:
#                 print("Not found: " + f_path)
#
#     if d == selected_species_list[3]:
#         for f in subfolder:
#             f_path = images_path + "/" + d + "/" + f
#             file_ = f
#             label = selected_species_list[3]
#             images_list.append([file_, label])
#             try:
#                 shutil.copy(f_path, images_dest)
#                 print("Moved: " + f_path)
#             except:
#                 print("Not found: " + f_path)
#
# image_df = pd.DataFrame(images_list, columns=columns, index=None)
# image_df.to_csv("../data/training/all_image_data.csv", index=False)
# print("Image files moved.")
#
# end = time.time()
#
# print("Took: " + str((end - start)) + "s")
# print("Done")
#
# # Multi
#
# img_df = pd.read_csv('../training_data/all_image_data.csv')
# aud_df = pd.read_csv('../training_data/all_audio_data.csv')
#
# audio_ = []
# all_imgs = []
# imgs_ = []
# labels_ = []
# final = []
#
# for l in selected_species_list:
#     aa = aud_df[aud_df['Label'] == l]["Species"].tolist()
#     ll = aud_df[aud_df['Label'] == l]["Label"].tolist()
#     audio_ += aa
#     labels_ += ll
#
# for a in selected_species_list:
#     ii = img_df[img_df['Label'] == a]["Species"].tolist()
#     all_imgs.append(ii)
#
# for i in range(0, 4):
#     imgs_ += all_imgs[i][0:43]
#
# for j, image in enumerate(imgs_):
#     aud_file = audio_[j].split(".wav")[0]
#     lbl = labels_[j]
#     final.append([image, aud_file + ".png", lbl])
#
# new_df = pd.DataFrame(final, columns=["Image", "Audio", "Label"], index=None)
# new_df.to_csv("../data/training/all_multi_data.csv", index=False)
# print("Done")
