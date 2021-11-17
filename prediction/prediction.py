from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import numpy as np
import cv2

audionet_model_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/audionet.h5"
image_model_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/resnet.h5"

audionet = load_model(audionet_model_path)
audionet.trainable = False
audionet2 = Model(audionet.inputs, audionet.layers[-2].output)

resnet = load_model(image_model_path)
resnet.trainable = False
resnet2 = Model(resnet.inputs, resnet.layers[-2].output)

print(resnet.summary())
print("*********************************************************")
print(resnet2.summary())

# Audio prediction
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

# Process mfcc
mfcc = mfcc.replace("[", "").replace("]", "")
mfcc = mfcc.replace("\n", "")
mfcc = mfcc.split()
mfcc = np.array(mfcc).astype(np.float32)
mfcc = mfcc.reshape(1, 40)

# Predict audio
a1_pred = audionet.predict(mfcc.reshape(1, 40))

# Extract audio features
a2_pred = audionet2.predict(mfcc.reshape(1, 40))

# Print prediction and features
print("audionet1 pred: ", a1_pred)
print("audionet2 features: ", a2_pred.shape)

###################################################################################
img_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/all/images/0070.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (100, 100), cv2.INTER_LINEAR)
img = img / 255.0
img = np.array(img)
img = img.reshape(1, 100, 100, 3)

img_pred = resnet.predict(img)
img_pred2 = resnet2.predict(img)
print("resnet pred: ", img_pred)
print("resnet2 features: ", img_pred2.shape)
