from models.model import *

# resnet, inception,efficient_net
INPUT_SHAPE = (150, 150, 3)
INPUT_TENSOR = Input(shape=INPUT_SHAPE)

scd = single_class_detector('resnet', INPUT_TENSOR, INPUT_SHAPE, 'imagenet')
base_output = scd.output
flatten = Flatten()(base_output)
det_head = detection_head1(flatten)

model = Model(inputs=scd.input, outputs=det_head)
