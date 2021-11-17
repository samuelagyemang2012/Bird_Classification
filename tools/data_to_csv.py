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
multi_data = []

for i, a in enumerate(audio_list):
    all_audio.append([a, audio_labels[i]])

for j, i in enumerate(images_list):
    all_images.append([i, images_labels[j], annotations_list[j]])

all_audio_df = pd.DataFrame(all_audio, columns=['file', 'class'], index=None)
# all_audio_df.to_csv("../data/all_audio_data.csv", index=False)

all_image_df = pd.DataFrame(all_images, columns=['file', 'class', 'bbox'], index=None)
# all_image_df.to_csv("../data/all_image_data.csv", index=False)



