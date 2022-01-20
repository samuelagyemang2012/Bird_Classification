from tensorflow.keras.optimizers import Adam
from evaluation.evaluation import *
from tensorflow.keras.utils import to_categorical
from models.model import *
from numpy.random import seed
import random
import tensorflow as tf

random.seed(89)
seed(25)
tf.random.set_seed(40)

EPOCHS = 180  # 200
INPUT_SHAPE = (40,)
BATCH_SIZE = 32
NUM_CLASSES = 3
VAL_SPLIT = 0.2

# Define paths
TRAIN_DATA_PATH = "../data/audio/train.csv"
TEST_DATA_PATH = "../data/audio/test.csv"
BEST_MODEL_PATH = "D:/Datasets/my_coco/trained_models/audionet.h5"
# "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/trained_models/audionet.h5"

TRAIN_LABELS = []
TEST_LABELS = []

TRUE_LABELS = ["bird", "dog", "car"]
LABELS = [0, 1, 2]

# Load data
print("Loading training data")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

train_ = train_df['mfcc'].tolist()
train_ = [np.array(t.replace("[", "").replace("]", "").replace("\n", "").split()).astype(np.float32) for t in train_]

test_ = test_df['mfcc'].tolist()
test_ = [np.array(ts.replace("[", "").replace("]", "").replace("\n", "").split()).astype(np.float32) for ts in test_]

train_labels_ = train_df['class'].tolist()
test_labels_ = test_df['class'].tolist()

for tl in train_labels_:
    if tl == 'bird':
        TRAIN_LABELS.append(0)

    if tl == 'dog':
        TRAIN_LABELS.append(1)

    if tl == 'car':
        TRAIN_LABELS.append(2)

for tt in test_labels_:
    if tt == 'bird':
        TEST_LABELS.append(0)

    if tt == 'dog':
        TEST_LABELS.append(1)

    if tt == 'car':
        TEST_LABELS.append(2)

# Normalize data
print("Convert data to numpy array")
TRAIN_DATA = np.array(train_)
TEST_DATA = np.array(test_)

print('Converting data to float')
TRAIN_DATA = TRAIN_DATA.astype('float32')
TEST_DATA = TEST_DATA.astype('float32')

# One-hot encode labels
print("One-hot encoding labels")
TRAIN_LABELS = to_categorical(TRAIN_LABELS)
TEST_LABELS = to_categorical(TEST_LABELS)

# Setup callbacks
print("Setting up callbacks")

callbacks = create_callbacks()  # BEST_MODEL_PATH + name + file_ext, "loss", "min", 5)

# Building model
print("Building model")
model = audio_net2(INPUT_SHAPE, NUM_CLASSES)
opts = Adam(learning_rate=0.0001)
model.compile(optimizer=opts, loss="categorical_crossentropy", metrics=['accuracy'])

# Train model
print("Training model")
history = model.fit(TRAIN_DATA, TRAIN_LABELS,
                    validation_data=(TEST_DATA, TEST_LABELS),
                    # callbacks=callbacks,
                    # steps_per_epoch=len(TRAIN_DATA) // BATCH_SIZE,
                    epochs=EPOCHS)

# Evaluate model
name = 'audio'
print("Evaluating model on " + str(len(TEST_DATA)) + " sounds")
acc = model.evaluate(TEST_DATA, TEST_LABELS, batch_size=BATCH_SIZE)
preds = model.predict(TEST_DATA, verbose=0)
preds = np.argmax(preds, axis=1)
model_loss_path = "../graphs/" + name + "_loss_mfcc.png"
model_acc_path = "../graphs/" + name + "_acc_mfcc.png"
model_cm_path = "../graphs/" + name + "_cm_mfcc.png"
plot_confusion_matrix(TEST_LABELS, preds, TRUE_LABELS, name, model_cm_path)
acc_loss_graphs_to_file(name, history, ['train', 'val'], 'upper left', model_loss_path, model_acc_path)
model_metrics_path = "../results/" + name + "_metrics_mfcc.txt"
metrics_to_file(name, model_metrics_path, TEST_LABELS, preds, TRUE_LABELS, acc)
model.save(BEST_MODEL_PATH)

print("Done!")
