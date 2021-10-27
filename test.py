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

path = ""
