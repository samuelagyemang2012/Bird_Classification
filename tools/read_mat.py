import scipy.io
import os
import cv2
from tools.pas_voc_to_custom import show_bbox

mat_path = "D:/Datasets/my_coco/cars_train_annos.mat"
image_base = "D:/Datasets/my_coco/images/car/images/"
ground_truth_path = "D:/Datasets/my_coco/images/car/ground_truths/"
annotations_path = "D:/Datasets/my_coco/all/annotations/car/"


def write(image_h, image_w, name, x1, y1, x2, y2, path):
    data = "<annotation><folder>02085620</folder><filename>n02085620_242</filename><source><database>Stanford AI " \
           "database</database></source><size> "
    data += "<width>" + str(image_w) + "</width>"
    data += "<height>" + str(image_h) + "</height>"
    data += "<depth>3</depth>"
    data += "</size>"
    data += "<segment> 0 </segment>"
    data += "<object>"
    data += "<name>" + name + "</name>"
    data += "<pose> Unspecified </pose>"
    data += "<truncated> 0 </truncated>"
    data += "<difficult> 0 </difficult>"
    data += "<bndbox>"
    data += "<xmin>" + str(x1) + "</xmin>"
    data += "<ymin>" + str(y1) + "</ymin>"
    data += "<xmax>" + str(x2) + "</xmax>"
    data += "<ymax>" + str(y2) + "</ymax>"
    data += "</bndbox>"
    data += "</object>"
    data += "</annotation>"

    f = open(path, "w")
    f.write(data)
    f.close()


imgs = os.listdir(image_base)
ann = []

mat = scipy.io.loadmat(mat_path)['annotations'][0]

for m in mat[0:1000]:
    x1 = m[0].item()
    y1 = m[1].item()
    x2 = m[2].item()
    y2 = m[3].item()

    ann.append([x1, y1, x2, y2])

for i, img in enumerate(imgs[0:1000]):
    name = img.split(".")[0]
    image = cv2.imread(image_base + img)
    h, w, c = image.shape
    x1, y1, x2, y2 = ann[i]

    print(x1, y1, x2, y2)

    write(h, w, name, x1, y1, x2, y2, annotations_path + name + ".xml")
    show_bbox(image, x1, y1, x2, y2, (0, 255, 0), 2, ground_truth_path + name + ".jpg")
    print(i)
