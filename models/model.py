from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, MaxPooling2D, Conv2D, BatchNormalization, \
    Concatenate
from tensorflow.keras.applications import ResNet50, InceptionV3, EfficientNetB4


# Resnet 50
def resnet_50(input_tensor, input_shape, weights):
    base = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


def inception(input_tensor, input_shape, weights):
    base = InceptionV3(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


def efficient_net(input_tensor, input_shape, weights):
    base = EfficientNetB4(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=input_shape)
    base.trainable = False
    return base


# Feed forward
def fully_connected(num_classes):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def multi_model(input_tensor1, input_tensor2, input_shape1, input_shape2, num_classes, weights):
    base1 = efficient_net(input_tensor1, input_shape1, weights)
    base1._name = 'audio_net'

    base2 = efficient_net(input_tensor2, input_shape2, weights)
    base2._name = 'image_net'

    concat = Concatenate(axis=1)([base1.output, base2.output])
    fc = fully_connected(num_classes)

    model1 = Sequential()
    model2 = Sequential()
    model = Sequential()

    m1 = model1.add(base1)
    m2 = model2.add(base2)

    model.add(m1)
    model.add(m2)
    model.add(concat)
    model.add(fc)

    return model


def build_model(base, forward):
    model = Sequential()
    model.add(base)
    model.add(forward)
    return model
