# import cv2
# from models.model import *
# from tensorflow.keras.utils import plot_model
#
# Test model
# path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/training_data/new_spectograms/Parus-major-168221.png"
# img = cv2.imread(path)
#
# INPUT_SHAPE = (150, 150, 3)
# input_tensor = Input(shape=INPUT_SHAPE)
#
# # model = multi_model(input_tensor, input_tensor, INPUT_SHAPE, INPUT_SHAPE, 4, 'imagenet')
# model = test_model()
#
#
# print(model.summary())
# plot_model(model, to_file='vis/model_plot.png', show_shapes=True, show_layer_names=True)
#######################################################################################################################

# Multi mover
# import pandas as pd
#
# img_df = pd.read_csv('../training_data/all_image_data.csv')
# aud_df = pd.read_csv('../training_data/all_audio_data.csv')
#
# labels = ["Fringilla coelebs", "Parus major", "Sylvia communis", "Turdus merula"]
#
# audio_ = []
# all_imgs = []
# imgs_ = []
# labels_ = []
# final = []
#
# for l in labels:
#     aa = aud_df[aud_df['Label'] == l]["Species"].tolist()
#     ll = aud_df[aud_df['Label'] == l]["Label"].tolist()
#     audio_ += aa
#     labels_ += ll
#
# for a in labels:
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
# new_df.to_csv("data/training/all_multi_data.csv", index=False)
# print("Done")

##############################################################
import cv2
# Bounding Box

from tools.pas_voc_to_yolo import convert_annotation
import cv2

convert_annotation("C:/Users/Administrator/Desktop/dd/labels/0009.xml",
                   "C:/Users/Administrator/Desktop/dcdc.txt",
                   ["bird"])

img_path = "C://Users/Administrator/Desktop/dd/images/0009.jpg"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# bbox = (0.416141235813367, 0.7029109589041096, 0.48675914249684743, 0.3613013698630137)
xy1 = (int(0.416141235813367 * w), int(0.7029109589041096 * h))
xy2 = (int(0.48675914249684743 * w), int(0.3613013698630137 * h))

x1y1 = (138, 306)  # (0.416141235813367, 0.7029109589041096)
x2y2 = (524, 517)  # (0.48675914249684743, 0.3613013698630137)

cv2.rectangle(img, xy1, xy2, (255, 0, 0), 2)
cv2.imshow("Output", img)
cv2.waitKey(0)
