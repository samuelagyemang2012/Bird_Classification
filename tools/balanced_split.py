import pandas as pd
import random

image_base = "G:/datasets/birdsounds/training_data/images"
image_path = "../data/training/all_image_data.csv"

TRAIN_SPLIT = 0.8

image_df = pd.read_csv(image_path)

image_files = image_df["Species"].tolist()

FC = image_df[image_df["Label"] == "Fringilla coelebs"]["Species"].tolist()
PM = image_df[image_df["Label"] == "Parus major"]["Species"].tolist()
TM = image_df[image_df["Label"] == "Turdus merula"]["Species"].tolist()
SC = image_df[image_df["Label"] == "Sylvia communis"]["Species"].tolist()

FC_train = FC[0:int(len(FC) * TRAIN_SPLIT)]
FC_train_labels = ["Fringilla coelebs" for i in range(int(len(FC) * TRAIN_SPLIT))]

PM_train = PM[0:int(len(PM) * TRAIN_SPLIT)]
PM_train_labels = ["Parus major" for i in range(int(len(PM) * TRAIN_SPLIT))]

TM_train = TM[0:int(len(TM) * TRAIN_SPLIT)]
TM_train_labels = ["Turdus merula" for i in range(int(len(TM) * TRAIN_SPLIT))]

SC_train = SC[0:int(len(SC) * TRAIN_SPLIT)]
SC_train_labels = ["Sylvia communis" for i in range(int(len(SC) * TRAIN_SPLIT))]

#########################################################################################

FC_test = FC[int(len(FC) * TRAIN_SPLIT):]
FC_test_labels = ["Fringilla coelebs" for i in range(len(FC) - int(len(FC) * TRAIN_SPLIT))]

PM_test = PM[int(len(PM) * TRAIN_SPLIT):]
PM_test_labels = ["Parus major" for i in range(len(PM) - int(len(PM) * TRAIN_SPLIT))]

TM_test = TM[int(len(TM) * TRAIN_SPLIT):]
TM_test_labels = ["Turdus merula" for i in range(len(TM) - int(len(TM) * TRAIN_SPLIT))]

SC_test = SC[int(len(SC) * TRAIN_SPLIT):]
SC_test_labels = ["Sylvia communis" for i in range(len(SC) - int(len(SC) * TRAIN_SPLIT))]

train_data = FC_train + PM_train + TM_train + SC_train
train_labels = FC_train_labels + PM_train_labels + TM_train_labels + SC_train_labels

test_data = FC_test + PM_test + TM_test + SC_test
test_labels = FC_test_labels + PM_test_labels + TM_test_labels + SC_test_labels

train_temp = list(zip(train_data, train_labels))
random.shuffle(train_temp)
train_data, train_labels = zip(*train_temp)

test_temp = list(zip(test_data, test_labels))
random.shuffle(test_temp)
test_data, test_labels = zip(*test_temp)

all_train_data = list(zip(train_data, train_labels))
all_test_data = list(zip(test_data, test_labels))

train_df = pd.DataFrame(all_train_data, columns=["Species", "Label"], index=None)
test_df = pd.DataFrame(all_test_data, columns=["Species", "Label"], index=None)

print("Train data")
print(train_df['Label'].value_counts())
print("")
print("Test data")
print(test_df['Label'].value_counts())

train_df.to_csv("../data/training/image/balanced_train.csv", index=False)
test_df.to_csv("../data/training/image/balanced_test.csv", index=False)
