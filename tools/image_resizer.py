import pandas as pd
import numpy as np
import cv2
from pas_voc_to_custom import get_bbox, show_bbox


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.
    Args:
        bbox (~numpy.ndarray): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`bbox`, ":math:`(R, 4)`", :obj:`float32`, \
        ":math:`(y_{min}, x_{min}, y_{max}, x_{max})`"
    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.
    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def resize_images_bbox(img_base, image_list, label_list, annotations, input_shape_, dest_path):
    data = []

    for i in range(len(image_list)):
        img = cv2.imread(img_base + image_list[i])
        bbox = annotations[i]
        h, w, c = img.shape
        y1, x1, y2, x2 = bbox[1], bbox[0], bbox[3], bbox[2]
        new_format = np.array([[y1, x1, y2, x2]])

        new_img = cv2.resize(img, input_shape_, cv2.INTER_LINEAR)
        cv2.imwrite(dest_path + image_list[i], new_img)
        new_bbox = resize_bbox(new_format, (h, w), input_shape_)
        new_x1, new_y1, new_x2, new_y2 = (new_bbox[0][1], new_bbox[0][0], new_bbox[0][3], new_bbox[0][2])

        data.append([image_list[i], new_x1, new_y1, new_x2, new_y2, label_list[i]])
        # show_bbox(new_img, new_x1, new_y1, new_x2, new_y2, (255, 0, 0), 1, "")

    return data
    # show_bbox(new_img, new_x1, new_y1, new_x2, new_y2, (255, 0, 0), 1, "")


image_base = "D:/Datasets/my_coco/all/actual/images/"
# annotations_base = "D:/Datasets/my_coco/all/annotations/"
dest_path = "D:/Datasets/my_coco/all/resized/images/"

bbox_columns = ["xmin", "ymin", "xmax", "ymax"]
input_shape = (300, 300)

# Load csv data
all_images_df = pd.read_csv('D:/Datasets/my_coco/_csvs/image_data.csv', index_col=None)
all_images_df['class'] = all_images_df['class'].map({'bird': 0, 'dog': 1, "car": 2})

# Preprocess annotations
all_ann_df = all_images_df[bbox_columns]
anns = all_ann_df.to_numpy().astype(int)

# Get image files
images = all_images_df['file'].to_list()

# Get labels
labels = all_images_df['class'].to_list()

# resize
train_data = resize_images_bbox(image_base,
                                images,
                                labels,
                                anns,
                                input_shape,
                                "D:/Datasets/my_coco/all/resized/images/")

data = pd.DataFrame(train_data, columns=["file", "xmin", "ymin", "xmax", "ymax", "class"])
data.to_csv("D:/Datasets/my_coco/_csvs/image_data_resized.csv", index=None)
