import pandas as pd
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from evaluation.evaluation import *
from tensorflow.keras.utils import to_categorical
from models.model import *
from numpy.random import seed
import random
import tensorflow as tf

random.seed(89)
seed(25)
tf.random.set_seed(40)

EPOCHS = 9
INPUT_SHAPE_1 = (100, 100, 3)
BATCH_SIZE = 16
NUM_CLASSES = 2
VAL_SPLIT = 0.1

# Define paths
IMG_BASE_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/all/images/"
MULTI_TRAIN_DATA_PATH = "../data/multi/train.csv"
MULTI_TEST_DATA_PATH = "../data/multi/test.csv"
BEST_MODEL_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/multimodal.h5"

TRUE_LABELS = ["bird", "dog"]
LABELS = [0, 1]

IMG_TRAIN_DATA = []
IMG_TEST_DATA = []
AUD_TRAIN_DATA = []
AUD_TEST_DATA = []

TRAIN_LABELS = []
TEST_LABELS = []

# Load data
print("Loading training data")
train_df = pd.read_csv(MULTI_TRAIN_DATA_PATH)
test_df = pd.read_csv(MULTI_TEST_DATA_PATH)

train_img = train_df['image'].tolist()
train_aud = train_df['mfcc'].tolist()

test_img = test_df['image'].tolist()
test_aud = test_df['mfcc'].tolist()

train_labels_ = train_df['class'].tolist()
test_labels_ = test_df['class'].tolist()

# Preprocess images
print("Resize train images")
for tr in range(len(train_img)):
    img = cv2.imread(IMG_BASE_PATH + train_img[tr])
    img_file = cv2.resize(img, (INPUT_SHAPE_1[0], INPUT_SHAPE_1[1]), cv2.INTER_LINEAR)
    IMG_TRAIN_DATA.append(img_file)

for trl in train_labels_:
    if trl == TRUE_LABELS[0]:
        TRAIN_LABELS.append(0)

    if trl == TRUE_LABELS[1]:
        TRAIN_LABELS.append(1)

print("Resize test images")
for tt in range(len(test_img)):
    img = cv2.imread(IMG_BASE_PATH + test_img[tt])
    img_file = cv2.resize(img, (INPUT_SHAPE_1[0], INPUT_SHAPE_1[1]), cv2.INTER_LINEAR)
    IMG_TEST_DATA.append(img_file)

for ttl in test_labels_:
    if ttl == TRUE_LABELS[0]:
        TEST_LABELS.append(0)

    if ttl == TRUE_LABELS[1]:
        TEST_LABELS.append(1)

print("Preprocess MFCCs")
AUD_TRAIN_DATA = [np.array(t.replace("[", "").replace("]", "").replace("\n", "").split()).astype(np.float32) for t in
                  train_aud]
AUD_TEST_DATA = [np.array(tt.replace("[", "").replace("]", "").replace("\n", "").split()).astype(np.float32) for tt in
                 test_aud]

# Normalize data
print("Normalizing data")
IMG_TRAIN_DATA = np.array(IMG_TRAIN_DATA)
AUD_TRAIN_DATA = np.array(AUD_TRAIN_DATA)

IMG_TEST_DATA = np.array(IMG_TEST_DATA)
AUD_TEST_DATA = np.array(AUD_TEST_DATA)

# Convert data to float
IMG_TRAIN_DATA = IMG_TRAIN_DATA.astype('float32')
AUD_TRAIN_DATA = AUD_TRAIN_DATA.astype('float32')

IMG_TEST_DATA = IMG_TEST_DATA.astype('float32')
AUD_TEST_DATA = AUD_TEST_DATA.astype('float32')

# One-hot encode labels
print("One-hot encoding labels")
TRAIN_LABELS = to_categorical(TRAIN_LABELS)
TEST_LABELS = to_categorical(TEST_LABELS)

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     horizontal_flip=True,
#     shear_range=0.2,
#     zoom_range=0.2,
#     validation_split=VAL_SPLIT,
# )
#
# test_datagen = ImageDataGenerator(
#     rescale=1. / 255,
# )
#
#
# # Data Augmentation
# def generator_data_fit(generator, first_data, second_data, labels, subset):
#     first_gen = generator.flow(first_data, labels, batch_size=BATCH_SIZE, subset=subset)
#     second_gen = generator.flow(second_data, labels, batch_size=BATCH_SIZE, subset=subset)
#
#     while True:
#         first_x = first_gen.next()
#         second_x = second_gen.next()
#
#         yield [first_x[0], second_x[0]], first_x[1]
#
#
# def generator_test_data_fit(generator, first_data, second_data, labels):
#     first_gen = generator.flow(first_data, labels, batch_size=BATCH_SIZE)
#     second_gen = generator.flow(second_data, labels, batch_size=BATCH_SIZE)
#
#     while True:
#         first_x = first_gen.next()
#         second_x = second_gen.next()
#
#         yield [first_x[0], second_x[0]], first_x[1]
#
#
# # print("Augmenting training data")
# # img_train_gen = train_datagen.flow(IMG_TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='training')
# # aud_train_gen = train_datagen.flow(AUD_TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='training')
# #
# # img_val_gen = train_datagen.flow(IMG_TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='validation')
# # aud_val_gen = train_datagen.flow(AUD_TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='validation')
# #
# # img_test_gen = test_datagen.flow(IMG_TEST_DATA)
# # aud_test_gen = test_datagen.flow(AUD_TEST_DATA)
#
# Setup callbacks

print("Setting up callbacks")
callbacks = create_callbacks()  # BEST_MODEL_PATH + name + file_ext, "loss", "min", 5)

# Building model
print("Building model")

audionet_model_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/audionet.h5"
image_model_path = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/resnet.h5"

audionet = load_model(audionet_model_path)
audionet.trainable = False
audionet = Model(audionet.inputs, audionet.layers[-2].output)

resnet = load_model(image_model_path)
resnet.trainable = False
resnet = Model(resnet.inputs, resnet.layers[-2].output)

# audionet = audio_net2((40,), 2)
# audionet = Model(audionet.inputs, audionet.layers[-2].output)
#
# input_tensor = Input(shape=INPUT_SHAPE_1)
# _base = resnet_50(input_tensor, INPUT_SHAPE_1, 'imagenet')
# flat1 = Flatten(name='Flatten_3')(_base.output)
# class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
# resnet = Model(inputs=_base.inputs, outputs=class1)

model = multi_model2(audionet, resnet, 2)
opts = Adam(learning_rate=0.0001)

model.compile(optimizer=opts, loss="categorical_crossentropy", metrics=['accuracy'])
print(model.summary())

# Train model
print("Training model")
history = model.fit([AUD_TRAIN_DATA, IMG_TRAIN_DATA], TRAIN_LABELS,
                    validation_data=([AUD_TEST_DATA, IMG_TEST_DATA], TEST_LABELS),
                    # callbacks=callbacks,
                    # validation_steps=int(len(VAL_SPLIT * IMG_TRAIN_DATA) // BATCH_SIZE),
                    # steps_per_epoch=int(len(IMG_TRAIN_DATA) // BATCH_SIZE),
                    epochs=EPOCHS)

# Evaluate model
name = 'multi'
print("Evaluating model")
acc = model.evaluate([AUD_TEST_DATA, IMG_TEST_DATA], TEST_LABELS, batch_size=BATCH_SIZE)
preds = model.predict([AUD_TEST_DATA, IMG_TEST_DATA], verbose=0)
preds = np.argmax(preds, axis=1)
model_loss_path = "../graphs/" + name + "_loss.png"
model_acc_path = "../graphs/" + name + "_acc.png"
model_cm_path = "../graphs/" + name + "_cm.png"
plot_confusion_matrix(TEST_LABELS, preds, TRUE_LABELS, name, model_cm_path)
acc_loss_graphs_to_file(name, history, ['train', 'val'], 'upper left', model_loss_path, model_acc_path)
model_metrics_path = "../results/" + name + "_metrics.txt"
metrics_to_file(name, model_metrics_path, TEST_LABELS, preds, TRUE_LABELS, acc)
model.save(BEST_MODEL_PATH)
print("Done!")
