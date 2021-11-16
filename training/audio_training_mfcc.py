import pandas as pd
import cv2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from evaluation.evaluation import *
from tensorflow.keras.utils import to_categorical
from models.model import *
from numpy.random import seed
import random
import tensorflow as tf

random.seed(89)
seed(25)
tf.random.set_seed(40)

EPOCHS = 10
INPUT_SHAPE = (40,)
BATCH_SIZE = 32
NUM_CLASSES = 2
VAL_SPLIT = 0.2

# Define paths
IMG_BASE_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/all/audio/"
TRAIN_DATA_PATH = "../data/audio/train1.csv"
TEST_DATA_PATH = "../data/audio/test2.csv"
# BEST_MODEL_PATH = "../trained_models/"

TRAIN_LABELS = []
TEST_LABELS = []

TRUE_LABELS = ["bird", "dog"]
LABELS = [0, 1]

# Load data
print("Loading training data")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

train_ = train_df['mfcc'].tolist()
train_ = [np.array(trd) for trd in train_]
# train_ = np.array(train_)

test_ = test_df['mfcc'].tolist()
test_ = [np.array(ttd) for ttd in test_]
test_ = np.array(test_)

train_labels_ = train_df['class'].tolist()
test_labels_ = test_df['class'].tolist()

for tl in train_labels_:
    if tl == 'bird':
        TRAIN_LABELS.append(0)
    else:
        TRAIN_LABELS.append(1)

for tt in test_labels_:
    if tt == 'bird':
        TEST_LABELS.append(0)
    else:
        TEST_LABELS.append(1)

print(type(train_[0]))
print(train_[0])
# Normalize data
# print("Normalizing data")
# TRAIN_DATA = np.array(TRAIN_DATA)
# TEST_DATA = np.array(TEST_DATA)
#
# TRAIN_DATA = TRAIN_DATA.astype('float32')
# TEST_DATA = TEST_DATA.astype('float32')
#

# # One-hot encode labels
# print("One-hot encoding labels")
# TRAIN_LABELS = to_categorical(TRAIN_LABELS)
# TEST_LABELS = to_categorical(TEST_LABELS)

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     validation_split=VAL_SPLIT,
# )
#
# test_datagen = ImageDataGenerator(
#     rescale=1. / 255,
# )
#
# # Data Augmentation
# print("Augmenting training data")
# train_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='training')
# val_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='validation')
# test_gen = test_datagen.flow(TEST_DATA, TEST_LABELS)
#
# # Setup callbacks
# print("Setting up callbacks")
#
# callbacks = create_callbacks()  # BEST_MODEL_PATH + name + file_ext, "loss", "min", 5)
#
# Building model
# print("Building model")
# input_tensor = Input(shape=INPUT_SHAPE)
# #
# model = audio_net(INPUT_SHAPE, 2)
# opts = Adam(learning_rate=0.0001)
# model.compile(optimizer=opts, loss="categorical_crossentropy", metrics=['accuracy'])
#
# print(len(train_.shape))
# print(len(TRAIN_LABELS))
# Train model
# print("Training model")
# history = model.fit(train_, TRAIN_LABELS,
#                     validation_data=(test_, TEST_LABELS),
#                     #                     # callbacks=callbacks,
#                     #                     # steps_per_epoch=len(TRAIN_DATA) // BATCH_SIZE,
#                     epochs=EPOCHS)
#
# # Evaluate model
# name = 'audio'
# print("Evaluating model on " + str(len(test_)) + " sounds")
# acc = model.evaluate(test_, TEST_LABELS, batch_size=BATCH_SIZE)
# preds = model.predict(test_, verbose=0)
# preds = np.argmax(preds, axis=1)
# model_loss_path = "../graphs/" + name + "_loss_mfcc.png"
# model_acc_path = "../graphs/" + name + "_acc_mfcc.png"
# model_cm_path = "../graphs/" + name + "_cm_mfcc.png"
# plot_confusion_matrix(TEST_LABELS, preds, TRUE_LABELS, name, model_cm_path)
# acc_loss_graphs_to_file(name, history, ['train', 'val'], 'upper left', model_loss_path, model_acc_path)
# model_metrics_path = "../results/" + name + "_metrics_mfcc.txt"
# metrics_to_file(name, model_metrics_path, TEST_LABELS, preds, TRUE_LABELS, acc)
#
# print("Done!")
