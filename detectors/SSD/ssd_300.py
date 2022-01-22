import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model
from detectors.configs import ssd_300_config as cfg


def resnet_():
    # # model_config
    # input_shape = (config.input_size, config.input_size, 3)
    # num_classes = len(label_map) + 1  # plus 1 for background
    # l2_reg = config.l2_reg
    # kernel_initializer = config.kernel_initializer
    # anchor_boxes_config = config.anchor_boxes_config
    # extra_box_for_ar_1 = config.extra_box_for_ar_1

    # get base vgg model as feature extractor
    base = ResNet50(
        input_shape=(100, 100, 3),
        classes=3,
        weights='imagenet',
        include_top=False
    )

    base.trainable = False
    base = Model(base.inputs, base.layers[-5].output)

    return base
    # base_model = Model(inputs=base_model.input,
    #                    outputs=base_model.get_layer('block5_conv3').output)
    #
    # """
    # block1_pool, 1_pool, 1 pool, pool1
    # block2_conv1, block2_1, conv2_1
    # block3_conv1, block3_1, conv3_1
    # """


# def detection_heads():

m = resnet_()
# plot_model(m)
print(m.summary())
#
# for l in m.layers:
#     print(l._name)
