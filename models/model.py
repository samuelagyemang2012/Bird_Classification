from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, MaxPooling2D, Conv2D, BatchNormalization, \
    Concatenate, Add, Activation
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB4


# Resnet 50
def resnet_50(input_tensor, input_shape, weights):
    base = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)

    for layer in base.layers:
        layer.trainable = False

    return base


def inception(input_tensor, input_shape, weights):
    base = InceptionV3(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)

    for layer in base.layers:
        layer.trainable = False

    return base


def efficient_net(input_tensor, input_shape, weights):
    base = EfficientNetB4(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False

    return base


def FC(num_classes):
    model = Sequential()
    model.add(Flatten())
    # model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def FC2(merge, num_classes):
    z = Flatten()(merge)
    z = Dense(1024, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(1024, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(num_classes, activation='softmax')(z)
    return z


def FC3(model, num_classes):
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(num_classes, activation='softmax')(class1)

    return output


def build_model(base, forward):
    model = Sequential()
    model.add(base)
    model.add(forward)
    return model


def multi_model(input_tensor1, input_tensor2, input_shape1, input_shape2, num_classes, weights):
    base1 = efficient_net(input_tensor1, input_shape1, weights)
    base1._name = 'audio_net'

    base2 = efficient_net(input_tensor2, input_shape2, weights)
    base2._name = 'image_net'

    x = base1(input_tensor1)
    x = Model(inputs=input_tensor1, outputs=x)

    y = base2(input_tensor2)
    y = Model(inputs=input_tensor2, outputs=y)

    merge = Add()([x.output, y.output])
    fc = FC2(merge, num_classes)

    model = Model(inputs=[x.input, y.input], outputs=fc)

    return model


def audio_net(input_shape, num_classes):
    model = Sequential()
    # first layer
    model.add(Dense(100, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # second layer
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # third layer
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # final layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def audio_net2(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(1024, input_shape=input_shape, activation="relu"))
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation="relu"))  # 256
    # model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # model.add(Activation('softmax'))

    return model


# -------------------------------------------------------------------------------------------------------------------
# Single class detector
def detection_head1(flatten):
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)

    return bboxHead


def single_class_detector(model_base, input_tensor, input_shape, weights):
    if model_base == "resnet":
        base = resnet_50(input_tensor, input_shape, weights)
        base.trainable = False
        return base
    elif model_base == "inception":
        base = inception(input_tensor, input_shape, weights)
        base.trainable = False
        return base
    elif model_base == "efficient_net":
        base = efficient_net(input_tensor, input_shape, weights)
        base.trainable = False
        return base
    else:
        print("model base not available")
        return

    # base_output = base.output
    # flatten = Flatten()(base_output)
    #
    # # construct a fully-connected layer header to output the predicted
    # # bounding box coordinates
    # det_head = detection_head1(flatten)
    #
    # # construct the model we will fine-tune for bounding box regression
    # model = Model(inputs=base.input, outputs=bboxHead)
