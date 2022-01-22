import pandas as pd

data_path = "D:/Datasets/my_coco/_csvs/image_data_resized.csv"

df = pd.read_csv(data_path)

image_df = df[['file', 'xmin', 'xmax', 'ymin', 'ymax', 'class']]
image_df = image_df.sample(frac=1).reset_index(drop=True)

# image_df['class'] = image_df['class'].map({'bird': 0, 'dog': 1, "car": 2})

train_size = int(0.7 * len(image_df))
val_size = int(0.2 * len(image_df))
test_size = int(0.1 * len(image_df))

train = image_df[0:train_size]
val = image_df[train_size:train_size + val_size]
test = image_df[train_size + val_size:]

# train['file'] = train['file'].str.split(".").str[0]
# val['file'] = val['file'].str.split(".").str[0]
# test['file'] = test['file'].str.split(".").str[0]

print("train:", len(train))
train.to_csv("D:/Datasets/my_coco/all/resized/detection_split/train.csv", index=None)
print(train['class'].value_counts())
print("")

print("val:", len(val))
val.to_csv("D:/Datasets/my_coco/all/resized/detection_split/val.csv", index=None)
print(val['class'].value_counts())
print("")

trainval = pd.concat([train, val])
print("trainval:", len(trainval))
trainval.to_csv("D:/Datasets/my_coco/all/resized/detection_split/trainval.csv", index=None)
print(trainval['class'].value_counts())
print("")

print("test:", len(test))
test.to_csv("D:/Datasets/my_coco/all/resized/detection_split/test.csv", index=None)
print(test['class'].value_counts())

print("sum: ", len(train) + len(val) + len(test))
