import pandas as pd
import os

SPLIT_SIZE = 0.8

image_data_path = "../data/all_image_data.csv"
audio_data_path = "../data/all_mfcc.csv"

image_df = pd.read_csv(image_data_path)
audio_df = pd.read_csv(audio_data_path)

x = audio_df[["class", "mfcc"]]
x['image'] = image_df['file']

x_df = pd.DataFrame(x, index=None, columns=['class', 'mfcc', 'image'])
x_df = x_df.sample(frac=1, axis=0).reset_index(drop=True)
x_df.to_csv('../data/all_multi.csv')

train_df = x_df[0:int(SPLIT_SIZE * len(x_df))]
train_df.to_csv("../data/multi/train.csv")
print("Train")
print(train_df['class'].value_counts())
print("-----------------")

test_df = x_df[int(SPLIT_SIZE * len(x_df)):]
test_df.to_csv("../data/multi/test.csv")
print("Test")
print(test_df['class'].value_counts())

