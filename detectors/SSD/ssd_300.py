import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from detectors.configs import ssd_300_config as cfg


def VGG(config, label_map, ):
    # model_config
    input_shape = (config.input_size, config.input_size, 3)
    num_classes = len(label_map) + 1  # plus 1 for background
    l2_reg = config.l2_reg
    kernel_initializer = config.kernel_initializer
    anchor_boxes_config = config.anchor_boxes_config
    extra_box_for_ar_1 = config.extra_box_for_ar_1

    # get base vgg model as feature extractor
    base_model = VGG16(
        input_shape=input_shape,
        classes=num_classes,
        weights='imagenet',
        include_top=False
    )

    base_model = Model(inputs=base_model.input,
                       outputs=base_model.get_layer('block5_conv3').output)

    """
    block1_pool, 1_pool, 1 pool, pool1
    block2_conv1, block2_1, conv2_1
    block3_conv1, block3_1, conv3_1
    """

    for layer in base_model.layers:
        if 'pool' in layer.name:
            new_name = layer.name.replace("block", "")
            new_name = new_name.split("_")
            new_name = new_name[1] + "" + new_name[0]
        else:
            new_name = layer.name.replace("conv", "")
            new_name = new_name.replace("block", "conv")

        base_model.get_layer(layer.name)._name = new_name
        base_model.get_layer(layer.name)._kernel_initializer = "he_normal"
        base_model.get_layer(layer.name).kernel_regularizer = l2(l2_reg)
        layer.trainable = False

    return base_model


m = VGG(cfg, [1, 2, 3])

for l in m.layers:
    print(l._name)
