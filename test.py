# # import cv2
# # from models.model import *
# # from tensorflow.keras.utils import plot_model
# #
# # Test model
# # path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/training_data/new_spectograms/Parus-major-168221.png"
# # img = cv2.imread(path)
# #
# # INPUT_SHAPE = (150, 150, 3)
# # input_tensor = Input(shape=INPUT_SHAPE)
# #
# # # model = multi_model(input_tensor, input_tensor, INPUT_SHAPE, INPUT_SHAPE, 4, 'imagenet')
# # model = test_model()
# #
# #
# # print(model.summary())
# # plot_model(model, to_file='vis/model_plot.png', show_shapes=True, show_layer_names=True)
# #######################################################################################################################
#
# # Multi mover
# # import pandas as pd
# #
# # img_df = pd.read_csv('../training_data/all_image_data.csv')
# # aud_df = pd.read_csv('../training_data/all_audio_data.csv')
# #
# # labels = ["Fringilla coelebs", "Parus major", "Sylvia communis", "Turdus merula"]
# #
# # audio_ = []
# # all_imgs = []
# # imgs_ = []
# # labels_ = []
# # final = []
# #
# # for l in labels:
# #     aa = aud_df[aud_df['Label'] == l]["Species"].tolist()
# #     ll = aud_df[aud_df['Label'] == l]["Label"].tolist()
# #     audio_ += aa
# #     labels_ += ll
# #
# # for a in labels:
# #     ii = img_df[img_df['Label'] == a]["Species"].tolist()
# #     all_imgs.append(ii)
# #
# # for i in range(0, 4):
# #     imgs_ += all_imgs[i][0:43]
# #
# # for j, image in enumerate(imgs_):
# #     aud_file = audio_[j].split(".wav")[0]
# #     lbl = labels_[j]
# #     final.append([image, aud_file + ".png", lbl])
# #
# # new_df = pd.DataFrame(final, columns=["Image", "Audio", "Label"], index=None)
# # new_df.to_csv("data/training/all_multi_data.csv", index=False)
# # print("Done")
#
# ##############################################################
# import cv2
# # Bounding Box
#
# # from tools.pas_voc_to_yolo import convert_annotation
# import cv2
#
# # convert_annotation("C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/images/train/0030.xml",
# #                    "C:/Users/Administrator/Desktop/dcdc.txt",
# #                    ["bird"])
#
# # img_path = "C://Users/Administrator/Desktop/dd/images/0009.jpg"
# # img = cv2.imread(img_path)
# # h, w = img.shape[:2]
# #
# # # bbox = (0.416141235813367, 0.7029109589041096, 0.48675914249684743, 0.3613013698630137)
# # xy1 = (int(0.416141235813367 * w), int(0.7029109589041096 * h))
# # xy2 = (int(0.48675914249684743 * w), int(0.3613013698630137 * h))
# #
# # x1y1 = (138, 306)  # (0.416141235813367, 0.7029109589041096)
# # x2y2 = (524, 517)  # (0.48675914249684743, 0.3613013698630137)
# #
# # cv2.rectangle(img, xy1, xy2, (255, 0, 0), 2)
# # cv2.imshow("Output", img)
# # cv2.waitKey(0)

########################################################################################################
# Draw bounding boxes

# import os
# from tools.pas_voc_to_custom import get_bbox, show_bbox
# import cv2
#
# images_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/images/"
# annotations_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/annotations/"
# bbx_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/GT/"
#
# images = os.listdir(images_path)
#
# for i in images:
#     name = i.split(".")[0]
#
#     img = cv2.imread(images_path + i)
#     x1, y1, x2, y2 = get_bbox(annotations_path + name)
#
#     show_bbox(img, x1, y1, x2, y2, (0, 255, 0), 2, bbx_path + name + ".jpg")

#######################################################################################################
# Move annotations from main dataset
# import os
# import shutil
#
# img_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/images/"
# annotation_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/annotations/"
# dataset_path = "C:/Users/Administrator/Desktop/standford_dogs/annotations/Annotation/"
#
# folders = os.listdir(dataset_path)
# files = os.listdir(img_path)
#
# for i, ff in enumerate(files):
#     data = ff.split("_")
#     folder = data[0]
#
#     for f in folders:
#         if folder == f.split("-")[0]:
#             annotation = dataset_path + f + "/" + ff.split(".")[0]
#             shutil.copy(annotation, annotation_path)

#####################################################################################
# import pandas as pd
# import shutil
#
# path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/dog_bark.csv"
# dataset_path = "C:/Users/Administrator/Desktop/archive/"
# s_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/audio/wav_files/"
#
# df = pd.read_csv(path)
#
# x = df[["slice_file_name", "fold"]]
#
# files = x["slice_file_name"].tolist()
# folders = x["fold"].tolist()
#
# c = 0
# for i, z in enumerate(files):
#     full_path = dataset_path + "fold" + str(folders[i]) + "/" + z
#     shutil.copy(full_path, s_path)
#
# print(c)


# import os
#
# audio_source = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/audio/wav_files/"
# audio_dest = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/dogs/audio/clipped_wav/"
#
# data = os.listdir(audio_source)
#
# for d in data:
#
#     input_path = audio_source + d
#     output_path = audio_dest + d
#     cmd = "ffmpeg -i " + input_path + " -ss 00:00:00 -t 00:00:01 " + output_path
#     os.system(cmd)
#
# print("done")

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import numpy as np

audionet_model_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/audionet.h5"
image_model_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/resnet.h5"

audionet = load_model(audionet_model_path)
resnet = load_model(image_model_path)

# extract = Model(model.inputs, model.layers[-3].output) # Dense(128,...)
# features = extract.predict(data)
print("before")
print(audionet.summary())
layers = audionet.layers[0: len(audionet.layers) - 1]

print("after")
audionet2 = Model(audionet.inputs, audionet.layers[-2].output)
print(audionet2.summary())

mfcc = '''[-4.9353143e+02  1.5473961e+02  8.1852407e+00  4.2252598e+00
  1.4097309e+01  1.6809113e+01  4.7005630e+00  1.1184429e+01
  5.8361278e+00  1.3143874e+01  1.1879518e+01  1.6093502e+01
  1.7254227e+00  5.4016829e+00  1.3161260e+00  5.3889580e+00
 -3.1956949e+00  5.0678639e+00  3.0815611e+00  4.4254055e+00
  1.3268406e+00  7.5907140e+00  1.8249539e+00  5.4979773e+00
 -2.3596996e-02  3.0480835e+00 -6.0524902e+00 -3.8193123e+00
 -2.8671663e+00  5.4725003e+00 -3.9062312e-01  2.7307472e-01
 -5.3830819e+00  1.3945873e+00  4.8293778e-01  2.5882666e+00
 -5.3621507e+00 -1.8952858e+00 -4.5658431e+00  6.1807925e-01]'''

mfcc = mfcc.replace("[", "").replace("]", "")
mfcc = mfcc.replace("\n", "")
mfcc = mfcc.split()
mfcc = np.array(mfcc).astype(np.float32)
mfcc = mfcc.reshape(1, 40)

a1_pred = audionet.predict(mfcc.reshape(1, 40))
a2_pred = audionet2.predict(mfcc.reshape(1, 40))

print("audionet1 pred: ", a1_pred)
print("audionet2 features: ", a2_pred)
