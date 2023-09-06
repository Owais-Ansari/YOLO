






# YOLO
Documentations: to-do/
availaible models : YOLOv3, YOLOv3_aspp/
availaible backbones : Darknet53, convnext_tiny, convnext_small/
availaible augmentations : {Geometric: [flip, Sharpen, RandomRotate, GaussNoise],/
                            color:[RGBShift ,HueSaturationValue, ColorJitter,RandomBrightnessContrast,ChannelShuffle]}/
augmentation: mosaic (to-do), cutmix(to-do)/
detect.py: to-do  (soon)/
bboxes_from_masks: to-do/
pretrained-link: to-do/
architectures to-do: yolov4, yolov5 (soon)/



# To use this repo update the config
```
import torch

DATASET = './PASCAL_VOC/'
#DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
GPU = 0
seed = 42#seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 8
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 700
CLIP = None#0.99999
CONF_THRESHOLD = 0.2
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
BACKBONE_FREEZE = False
PIN_MEMORY = True
RESUME = ''
SAVE_MODEL = True
CHECKPOINT_DIR = "exp1"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
STEP_SIZE = 300  #epochs after lr linear decay by 0.1
ACCUM_FACTOR = 1  # to turn on use 2,4, or more
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

```

# For trainig run below command in the terminal
python train.py
