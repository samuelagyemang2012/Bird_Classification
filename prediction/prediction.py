from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

audionet_model_path = "D:/Datasets/my_coco/trained_models/audionet.h5"
image_model_path = "D:/Datasets/my_coco/trained_models/resnet.h5"
multi_model_path = "D:/Datasets/my_coco/trained_models//multimodal.h5"

audionet = load_model(audionet_model_path)
audionet.trainable = False
audionet2 = Model(audionet.inputs, audionet.layers[-2].output)

resnet = load_model(image_model_path)
resnet.trainable = False
resnet2 = Model(resnet.inputs, resnet.layers[-2].output)

multi_model = load_model(multi_model_path)

labels = ["bird", "dog", "car"]


def preprocess_mfcc(mfcc_):
    mfcc = mfcc_.replace("[", "").replace("]", "")
    mfcc = mfcc.replace("\n", "")
    mfcc = mfcc.split()
    mfcc = np.array(mfcc).astype(np.float32)
    mfcc = mfcc.reshape(1, 40)

    return mfcc


def preprocess_img(img_path_):
    image = cv2.imread(img_path_)
    image = cv2.resize(image, (100, 100), cv2.INTER_LINEAR)
    image = image / 255.0
    image = np.array(image)
    image = image.reshape(1, 100, 100, 3)

    return image


print("*********************************************************")

# Single audio and image prediction
audio_mfcc = '''[-3.27780884e+02  1.21065681e+02 -4.00812073e+01 -4.06821175e+01
 -3.77676697e+01 -1.91292000e+01 -2.07194901e+01 -2.42134333e+00
 -2.60565434e+01  1.09487743e+01  1.23772955e+01  7.53735352e+00
  8.41231823e+00  6.79377794e+00 -6.95381212e+00 -1.46880264e+01
 -1.23672085e+01 -2.80073333e+00 -5.12733746e+00 -3.58712602e+00
 -1.75722033e-01  7.85280323e+00  7.23448658e+00  2.58814740e+00
 -7.58885288e+00  4.16145563e+00  7.98601580e+00  2.60556769e+00
 -1.35454893e+00  1.06728697e+00 -2.02247739e+00  2.17973781e+00
  2.66387272e+00  2.81490564e+00 -2.13995978e-01  4.38734554e-02
 -2.52553010e+00 -3.41716957e+00 -2.28755403e+00 -5.28283834e-01]'''

img_path = "D:/Datasets/my_coco/all/images/n02085782_82.jpg"

# Process mfcc and image
prep_mfcc = preprocess_mfcc(audio_mfcc)
prep_img = preprocess_img(img_path)

# Predict audio
audio_pred = audionet.predict(prep_mfcc)
img_pred = resnet.predict(prep_img)
multi_pred = multi_model.predict([prep_mfcc, prep_img])

# Print prediction and features
print("Single Prediction")
print("audionet pred: ", labels[audio_pred.argmax()])
print("confidence: ", round((audio_pred.max() * 100), 2))
print("")
print("resnet pred: ", labels[img_pred.argmax()])
print("confidence: ", round((img_pred.max() * 100), 2))
print("")
print("multi pred: ", labels[multi_pred.argmax()])
print("confidence: ", round((multi_pred.max() * 100), 2))
print("")

# Batch prediction
print("Batch Prediction")
df = pd.read_csv('../data/multi/test.csv')

mfccs = df['mfcc'].tolist()
images = df['image'].tolist()
true_labels = df['class'].tolist()
img_base = "D:/Datasets/my_coco/all/images/"

total1 = 0
total2 = 0
total3 = 0

for i, m in tqdm(enumerate(mfccs)):
    m = preprocess_mfcc(m)

    image_path = img_base + images[i]
    img = preprocess_img(image_path)

    multi_pred = multi_model.predict([m, img])
    img_pred = resnet.predict(img)
    a1_pred = audionet.predict(m)

    m_pred = labels[multi_pred.argmax()]
    m_conf = round((multi_pred.max() * 100), 2)

    i_pred = labels[img_pred.argmax()]
    i_conf = round((img_pred.max() * 100), 2)

    a_pred = labels[a1_pred.argmax()]
    a_conf = round((a1_pred.max() * 100), 2)

    true_label = true_labels[i]

    if m_pred == true_label and m_conf > 50.0:
        total1 += 1

    if i_pred == true_label and i_conf > 50.0:
        total2 += 1

    if a_pred == true_label and a_conf > 50.0:
        total3 += 1

print("audio score: " + str(total3) + "/" + str(len(true_labels)))
print("image score: " + str(total2) + "/" + str(len(true_labels)))
print("multi score: " + str(total1) + "/" + str(len(true_labels)))
