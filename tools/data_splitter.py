import pandas as pd
from sklearn.model_selection import train_test_split
import random

audio_base = "G:/datasets/birdsounds/training_data/audio/wav"
audio_path = "../data/training/all_audio_data.csv"

image_base = "G:/datasets/birdsounds/training_data/images"
image_path = "../data/training/all_image_data.csv"

TRAIN_SPLIT = 0.8

image_df = pd.read_csv(image_path)
audio_df = pd.read_csv(audio_path)

image_files = image_df["Species"].tolist()
image_labels = image_df["Label"].tolist()
print("Total Images: " + str(len(image_files)))

audio_files = audio_df["Species"].tolist()
audio_labels = audio_df["Label"].tolist()
print("Total Audio: " + str(len(audio_files)))


def split(data, labels, train_size, columns, random_state, train_path, test_path):
    # Shuffle data
    data_train = []
    data_test = []

    temp = list(zip(data, labels))
    random.shuffle(temp)
    data, labels = zip(*temp)

    # Split images
    # X_train, X_test, y_train, y_test
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels,
                                                                        train_size=train_size,
                                                                        random_state=random_state)
    for i in range(len(train_data)):
        data_train.append([train_data[i], train_labels[i]])

    for i in range(len(test_data)):
        data_test.append([test_data[i], test_labels[i]])

    train_df = pd.DataFrame(data_train, columns=columns, index=None)
    train_df.to_csv(train_path, index=False)

    test_df = pd.DataFrame(data_test, columns=columns, index=None)
    test_df.to_csv(test_path, index=False)

    print("Train data: " + str(len(train_data)))
    print("Test data: " + str(len(test_data)))
    print("------------------------")
    print(train_df[columns[1]].value_counts())


def balanced_split():
    pass


columns_ = ["Species", "Label"]

# Image split
split(image_files, image_labels, TRAIN_SPLIT, columns_, 200, "../data/training/image/train.csv",
      "../data/training/image/test.csv")

# Audio split
# split(audio_files, audio_labels, TRAIN_SPLIT, columns_, 12, "../data/training/audio/train.csv",
#            "../data/training/audio/test.csv")
