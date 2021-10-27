import cv2
from models.model import *
from keras.utils.vis_utils import plot_model


path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/training_data/new_spectograms/Parus-major-168221.png"
img = cv2.imread(path)

INPUT_SHAPE = (150, 150, 3)
input_tensor = Input(shape=INPUT_SHAPE)

# model = multi_model(input_tensor, input_tensor, INPUT_SHAPE, INPUT_SHAPE, 4, 'imagenet')
model = test_model()


print(model.summary())
plot_model(model, to_file='vis/model_plot.png', show_shapes=True, show_layer_names=True)