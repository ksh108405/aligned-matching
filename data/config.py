# config.py
import os.path
from .config_util import get_min_max_size, size_index, conv4_3_size

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300, SSD512, SSD1024 CONFIGS
VOC_NETWORK_SIZE = 300  # 300 or 512 or 1024
VOC_CONV4_3_SIZE = 0.01
VOC_MIN_MAX_SIZE = get_min_max_size(VOC_NETWORK_SIZE, conv4_3_size[str(VOC_CONV4_3_SIZE)])
VOC_HALF_SIZED = False

TT100K_NETWORK_SIZE = 512  # 300 or 512 or 1024
TT100K_CONV4_3_SIZE = 0.10
TT100K_MIN_MAX_SIZE = get_min_max_size(TT100K_NETWORK_SIZE, conv4_3_size[str(TT100K_CONV4_3_SIZE)])
TT100K_HALF_SIZED = False

COCO_NETWORK_SIZE = 300  # 300 or 512 or 1024
COCO_CONV4_3_SIZE = 0.07
COCO_MIN_MAX_SIZE = get_min_max_size(COCO_NETWORK_SIZE, conv4_3_size[str(COCO_CONV4_3_SIZE)])
COCO_HALF_SIZED = False

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [[38, 19, 10, 5, 3, 1], [64, 32, 16, 8, 4, 2, 1], [128, 64, 32, 16, 8, 4, 2, 1]][
        size_index[str(VOC_NETWORK_SIZE)]],
    'min_dim': [300, 512, 1024][size_index[str(VOC_NETWORK_SIZE)]],
    'steps': [[8, 16, 32, 64, 100, 300], [8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256, 512, 1024]][
        size_index[str(VOC_NETWORK_SIZE)]],
    'min_sizes': VOC_MIN_MAX_SIZE[0],
    'max_sizes': VOC_MIN_MAX_SIZE[1],
    'aspect_ratios': [[[2], [2, 3], [2, 3], [2, 3], [2], [2]], [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                      [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]][size_index[str(VOC_NETWORK_SIZE)]],
    'half_sized': VOC_HALF_SIZED,
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC'
}

tt100k = {
    'num_classes': 6,
    'lr_steps': (50000, 75000, 100000),
    'max_iter': 100000,
    'feature_maps': [[38, 19, 10, 5, 3, 1], [64, 32, 16, 8, 4, 2, 1], [128, 64, 32, 16, 8, 4, 2, 1]][
        size_index[str(TT100K_NETWORK_SIZE)]],
    'min_dim': [300, 512, 1024][size_index[str(TT100K_NETWORK_SIZE)]],
    'steps': [[8, 16, 32, 64, 100, 300], [8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256, 512, 1024]][
        size_index[str(TT100K_NETWORK_SIZE)]],
    'min_sizes': TT100K_MIN_MAX_SIZE[0],
    'max_sizes': TT100K_MIN_MAX_SIZE[1],
    'aspect_ratios': [[[2], [2, 3], [2, 3], [2, 3], [2], [2]], [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                      [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]][size_index[str(TT100K_NETWORK_SIZE)]],
    'half_sized': TT100K_HALF_SIZED,
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'TT100K'
}

coco = {
    'num_classes': 81,
    'lr_steps': (280000, 320000, 360000),
    'max_iter': 360000,
    'feature_maps': [[38, 19, 10, 5, 3, 1], [64, 32, 16, 8, 4, 2, 1], [128, 64, 32, 16, 8, 4, 2, 1]][
        size_index[str(COCO_NETWORK_SIZE)]],
    'min_dim': [300, 512, 1024][size_index[str(COCO_NETWORK_SIZE)]],
    'steps': [[8, 16, 32, 64, 100, 300], [8, 16, 32, 64, 128, 256, 512], [8, 16, 32, 64, 128, 256, 512, 1024]][
        size_index[str(COCO_NETWORK_SIZE)]],
    'min_sizes': COCO_MIN_MAX_SIZE[0],
    'max_sizes': COCO_MIN_MAX_SIZE[1],
    'aspect_ratios': [[[2], [2, 3], [2, 3], [2, 3], [2], [2]], [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                      [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]][size_index[str(COCO_NETWORK_SIZE)]],
    'half_sized': COCO_HALF_SIZED,
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO'
}
