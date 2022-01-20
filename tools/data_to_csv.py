import pandas as pd
import os
import shutil
import time

classes = ["bird", "dog", "car"]
DEST_PATH = "D:/Datasets/my_coco/_csvs/"

# Images
image_bird_path = "D:/Datasets/my_coco/images/bird/images/"
image_dog_path = "D:/Datasets/my_coco/images/dog/images/"
image_car_path = "D:/Datasets/my_coco/images/car/images/"

# Annotations
bird_annotations_path = "D:/Datasets/my_coco/images/bird/annotations/"
dog_annotations_path = "D:/Datasets/my_coco/images/dog/annotations/"
car_annotations_path = "D:/Datasets/my_coco/images/car/annotations/"

# Audio
audio_bird_path = "D:/Datasets/my_coco/audio/bird/clipped_wav/"
audio_dog_path = "D:/Datasets/my_coco/audio/dog/clipped_wav/"
audio_car_path = "D:/Datasets/my_coco/audio/car/clipped_wav/"

# Image data
image_bird_data = os.listdir(image_bird_path)
image_dog_data = os.listdir(image_dog_path)
image_car_data = os.listdir(image_car_path)

# Image annotations
bird_ann = os.listdir(bird_annotations_path)
dog_ann = os.listdir(dog_annotations_path)
car_ann = os.listdir(car_annotations_path)

# Audio_data
audio_bird_data = os.listdir(audio_bird_path)
audio_dog_data = os.listdir(audio_dog_path)
audio_car_data = os.listdir(audio_car_path)

images_list = image_bird_data + image_dog_data[0:945] + image_car_data
audio_list = audio_bird_data + audio_dog_data + audio_car_data

images_labels = ["bird" for a in range(len(image_bird_data))] + \
                ["dog" for b in range(len(image_dog_data[0:945]))] + \
                ["car" for c in range(len(image_car_data))]

annotations_list = bird_ann + dog_ann[0:945] + car_ann

audio_labels = ["bird" for aa in range(len(audio_bird_data))] + \
               ["dog" for bb in range(len(audio_dog_data))] + \
               ["car" for cc in range(len(audio_car_data))]

print("image_data", len(images_list))
print("image_labels", len(images_labels))

print("audio_data", len(audio_list))
print("audio_labels", len(audio_labels))

print("annotations", len(annotations_list))

all_images = []
all_audio = []

for j, i in enumerate(images_list):
    all_images.append([i, images_labels[j], annotations_list[j]])
    all_audio.append([audio_list[j], audio_labels[j]])

all_image_df = pd.DataFrame(all_images, columns=['file', 'class', 'bbox'], index=None)
all_audio_df = pd.DataFrame(all_audio, columns=['file', 'class'], index=None)

all_image_df.to_csv(DEST_PATH + 'image_data.csv', index=False)
all_audio_df.to_csv(DEST_PATH + 'audio_data.csv', index=False)

print("Done")
