import xml.etree.ElementTree as ET
import os
import cv2
import pandas as pd

SOURCE_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/images/val/"
SUBSET = "val"
DETECTOR_PATH = "../detector/bboxes/"

IMAGES = []
BBOX = []
DATA = []
CLASS = "bird"


def get_bbox(xml_file):
    file_data = open(xml_file)
    tree = ET.parse(file_data)
    root = tree.getroot()
    object_ = root.find('object')
    bbx = object_.find('bndbox')
    x1 = bbx.find('xmin').text
    y1 = bbx.find('ymin').text
    x2 = bbx.find('xmax').text
    y2 = bbx.find('ymax').text

    return int(x1), int(y1), int(x2), int(y2)


def show_bbox(image, x1, y1, x2, y2, color, thickness, output_path):
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if output_path == '':
        cv2.imshow("Output", image)
        cv2.waitKey(0)
    else:
        cv2.imwrite(output_path, image)


def data_to_csv(data, output_file):
    df = pd.DataFrame(data, index=None, columns=["file", "X1", "Y1", "X2", "Y2", "class"])
    df.to_csv(output_file, index=False)


# for file in os.listdir(SOURCE_PATH):
#     if file.endswith(".jpg") or file.endswith(".png"):
#         IMAGES.append(file)
#
#     if file.endswith(".xml"):
#         x1, y1, x2, y2 = get_bbox(SOURCE_PATH + file)
#         BBOX.append((x1, y1, x2, y2))
#
# for i in range(len(BBOX)):
#     DATA.append([IMAGES[i], BBOX[i][0], BBOX[i][1], BBOX[i][2], BBOX[i][3], CLASS])
#
# if not os.path.isdir(DETECTOR_PATH + CLASS):
#     os.mkdir(DETECTOR_PATH + CLASS)
#     data_to_csv(DATA, DETECTOR_PATH + CLASS + "/" + SUBSET + ".csv")
# else:
#     data_to_csv(DATA, DETECTOR_PATH + CLASS + "/" + SUBSET + ".csv")
