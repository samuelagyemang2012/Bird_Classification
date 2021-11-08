import xml.etree.ElementTree as ET
import os

SOURCE_PATH = "C:/Users/Administrator/Desktop/Sam/Multimodal_Fusion/my_coco/images/val/"
DEST_PATH = "../detector/"

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

    return x1, y1, x2, y2


for file in os.listdir(SOURCE_PATH):

    if file.endswith(".jpg") or file.endswith(".png"):
        IMAGES.append(file)

    if file.endswith(".xml"):
        x1, y1, x2, y2 = get_bbox(SOURCE_PATH + file)
        BBOX.append((x1, y1, x2, y2))

for i in range(len(BBOX)):
    DATA.append([IMAGES[i], BBOX[i][0], BBOX[i][1], BBOX[i][2], BBOX[i][3], CLASS])

print(DATA)
