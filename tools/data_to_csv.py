import pandas as pd
import os
import shutil
import time

classes = ["bird", "dog", "car"]
DEST_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/data_csv/"

# Images
image_bird_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/images/bird/images/"
image_dog_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/images/dog/images/"
image_car_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/images/car/images/"

# Annotations
# bird_annotations_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/bird/annotations/"
# dog_annotations_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dog/annotations/"

# Image data
image_bird_data = os.listdir(image_bird_path)
image_dog_data = os.listdir(image_dog_path)
image_car_data = os.listdir(image_car_path)

# Image annotations
# bird_ann = os.listdir(bird_annotations_path)
# dog_ann = os.listdir(dog_annotations_path)


images_list = image_bird_data + image_dog_data[0:945] + image_car_data

images_labels = ["bird" for a in range(len(image_bird_data))] + \
                ["dog" for b in range(len(image_dog_data[0:945]))] + \
                ["car" for c in range(len(image_car_data))]

# annotations_list = bird_ann + dog_ann[0:945]


print("image_data", len(images_list))
print("image_labels", len(images_labels))
# print("annotations", len(annotations_list))

all_images = []

for j, i in enumerate(images_list):
    all_images.append([i, images_labels[j]])  # , annotations_list[j]])

all_image_df = pd.DataFrame(all_images, columns=['file', 'class'], index=None)
all_image_df.to_csv("../data/all_image_data.csv", index=False)
