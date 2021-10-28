from tensorflow.keras.models import Sequential, Model
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


def fully_connected2(concat, num_classes):
    z = Flatten()(concat)
    z = Dense(1024, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(1024, activation='relu')(z)
    z = Dropout(0.2)(z)
    z = Dense(num_classes, activation='softmax')(z)

    return z


# def test_model():
#     inputA = Input(shape=(32,))
#     inputB = Input(shape=(128,))
#     # the first branch operates on the first input
#     x = Dense(8, activation="relu")(inputA)
#     x = Dense(4, activation="relu")(x)
#     x = Model(inputs=inputA, outputs=x)

#     # the second branch opreates on the second input
#     y = Dense(64, activation="relu")(inputB)
#     y = Dense(32, activation="relu")(y)
#     y = Dense(4, activation="relu")(y)
#     y = Model(inputs=inputB, outputs=y)

#     # combine the output of the two branches
#     combined = Concatenate(axis=1)([x.input, y.input])

#     # apply a FC layer and then a regression prediction on the
#     # combined outputs
#     z = Dense(2, activation="relu")(combined)
#     z = Dense(1, activation="linear")(z)

#     # our model will accept the inputs of the two branches and
#     # then output a single value
#     model = Model(inputs=[x.input, y.input], outputs=z)
#     return model


def multi_model(input_tensor1, input_tensor2, input_shape1, input_shape2, num_classes, weights):
    base1 = efficient_net(input_tensor1, input_shape1, weights)
    base1._name = 'audio_net'

    base2 = efficient_net(input_tensor2, input_shape2, weights)
    base2._name = 'image_net'

    x = base1(input_tensor1)
    x = Model(inputs=input_tensor1, outputs=x)

    y = base2(input_tensor2)
    y = Model(inputs=input_tensor2, outputs=y)

    concat = Concatenate()([x.output, y.output])
    fc = fully_connected2(concat, num_classes)

    model = Model(inputs=[x.input, y.input], outputs=fc)

    return model


def build_model(base, forward):
    model = Sequential()
    model.add(base)
    model.add(forward)
    return model
