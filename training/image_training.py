import cv2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import GlobalAveragePooling2D, MaxPooling2D

from evaluation.evaluation import *
from tensorflow.keras.utils import to_categorical
from models.model import *
from numpy.random import seed
import random
import tensorflow as tf

random.seed(89)
seed(25)
tf.random.set_seed(40)

EPOCHS = 20
INPUT_SHAPE = (100, 100, 3)
BATCH_SIZE = 16
NUM_CLASSES = 3
VAL_SPLIT = 0.2

# Define paths
IMG_BASE_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/all/images/"
TRAIN_DATA_PATH = "../data/image/train.csv"
TEST_DATA_PATH = "../data/image/test.csv"
BEST_MODEL_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/resnet.h5"

TRAIN_DATA = []
TRAIN_LABELS = []
TEST_DATA = []
TEST_LABELS = []

TRUE_LABELS = ["bird", "dog", "car"]
LABELS = [0, 1, 2]

# Load data
print("Loading training data")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

train_ = train_df['file'].tolist()
test_ = test_df['file'].tolist()

train_labels_ = train_df['class'].tolist()
test_labels_ = test_df['class'].tolist()

# Resize train images
print("Resize train images")
for tr in train_:
    img = cv2.imread(IMG_BASE_PATH + tr)
    img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]), cv2.INTER_LINEAR)
    TRAIN_DATA.append(img)

for trl in train_labels_:
    if trl == TRUE_LABELS[0]:
        TRAIN_LABELS.append(0)

    if trl == TRUE_LABELS[1]:
        TRAIN_LABELS.append(1)

    if trl == TRUE_LABELS[2]:
        TRAIN_LABELS.append(2)

# Resize test images
print("Resize test images")
for tt in test_:
    img = cv2.imread(IMG_BASE_PATH + tt)
    img = cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1]), cv2.INTER_LINEAR)
    TEST_DATA.append(img)

for ttl in test_labels_:
    if ttl == TRUE_LABELS[0]:
        TEST_LABELS.append(0)

    if ttl == TRUE_LABELS[1]:
        TEST_LABELS.append(1)

    if ttl == TRUE_LABELS[2]:
        TEST_LABELS.append(2)

# Normalize data
print("Normalizing data")
TRAIN_DATA = np.array(TRAIN_DATA)
TEST_DATA = np.array(TEST_DATA)

TRAIN_DATA = TRAIN_DATA.astype('float32')
TEST_DATA = TEST_DATA.astype('float32')

# One-hot encode labels
print("One-hot encoding labels")
TRAIN_LABELS = to_categorical(TRAIN_LABELS)
TEST_LABELS = to_categorical(TEST_LABELS)
#
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=VAL_SPLIT,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
)

# Data Augmentation
print("Augmenting training data")
train_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='training')
val_gen = train_datagen.flow(TRAIN_DATA, TRAIN_LABELS, batch_size=BATCH_SIZE, subset='validation')
test_gen = test_datagen.flow(TEST_DATA, TEST_LABELS)

# Setup callbacks
print("Setting up callbacks")

callbacks = create_callbacks()  # BEST_MODEL_PATH + name + file_ext, "loss", "min", 5)

# Building model
print("Building model")
input_tensor = Input(shape=INPUT_SHAPE)

model = cnn(INPUT_SHAPE, NUM_CLASSES)

# base_ = resnet_50(input_tensor, INPUT_SHAPE, 'imagenet')
# x = Dense(NUM_CLASSES, activation='softmax', name='dense_Q')(base_.output)
# model = Model(inputs=base_.input, outputs=x)

# x = MaxPooling2D(name='avg_pool_B')(x)
# x = Flatten(name='flatten_X')(x)
# x = Dense(1024, activation='relu', name='dense_Q')(x)
# x = Dropout(0.2, name='dropout_E')(x)
# x = Dense(256, activation='relu', name='dense_W')(x)
# # x = Dense(128, activation='relu', name='dense_R')(x)

opts = Adam(learning_rate=0.0001)
model.compile(optimizer=opts, loss="categorical_crossentropy", metrics=['accuracy'])

# Train model
print("Training model")
history = model.fit(train_gen,
                    validation_data=val_gen,
                    # callbacks=callbacks,
                    # steps_per_epoch=len(TRAIN_DATA) // BATCH_SIZE,
                    epochs=EPOCHS)

# Evaluate model
name = 'image'
print("Evaluating model on " + str(len(TEST_DATA)) + " images")
acc = model.evaluate(TEST_DATA, TEST_LABELS, batch_size=BATCH_SIZE)
preds = model.predict(TEST_DATA, verbose=0)
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
