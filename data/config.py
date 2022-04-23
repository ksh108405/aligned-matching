# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300, SSD512 CONFIGS
voc = {  # ENABLE AUGMENTATIONS!!
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),  # for 512
    'max_iter': 120000,  # for 512
    # 'feature_maps': [38, 19, 10, 5, 3, 1],  # for 300
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],  # for 512
    # 'min_dim': 300,  # for 300
    'min_dim': 512,  # for 512
    # 'steps': [8, 16, 32, 64, 100, 300],  # for 300
    'steps': [8, 16, 32, 64, 128, 256, 512],  # for 512
    # 'min_sizes': [30, 60, 111, 162, 213, 264],  # for 300 (0.1, 0.2, 0.9)
    # 'max_sizes': [60, 111, 162, 213, 264, 315],  # for 300
    # 'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],  # for 512 (0.07, 0.15, 0.9)
    # 'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],  # for 512 (0.04, 0.10, 0.9)
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # for 300
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],  # for 512
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC'
}

tt100k = {  # DISABLE AUGMENTATIONS!! (ONLY ENABLE RESIZE, SUBTRACT_MEANS)
    'num_classes': 6,  # 전체 클래스 개수 5 + 백그라운드 클래스 1
    'lr_steps': (50000, 75000, 100000),  # for batch 32
    'max_iter': 100000,
    # 'feature_maps': [38, 19, 10, 5, 3, 1],  # for 300 (변경시 get_detected_info 확인)
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],  # for 512 (변경시 get_detected_info 확인)
    # 'min_dim': 300,
    'min_dim': 512,  # for 512
    # 'steps': [8, 16, 32, 64, 100, 300],  # for 300
    'steps': [8, 16, 32, 64, 128, 256, 512],  # for 512
    # 'min_sizes': [30, 60, 111, 162, 213, 264],  # for 300
    # 'max_sizes': [60, 111, 162, 213, 264, 315],  # for 300
    # 'min_sizes': [51.2, 102.4, 174.08, 245.76, 317.44, 389.12, 460.8],  # for 512 (0.1, 0.2, 0.9)
    # 'max_sizes': [102.4, 174.08, 245.76, 317.44, 389.12, 460.8, 532.48],  # for 512
    # 'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],  # for 512 (0.07, 0.15, 0.9)
    # 'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],  # for 512 (0.04, 0.10, 0.9)
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # for 300 (변경시 get_detected_info 확인)
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],  # for 512 (변경시 get_detected_info 확인)
    # 'aspect_ratios': [[2], [2], [2], [2], [2], [2], [2]],
    # 'aspect_ratios': [[], [], [], [], [], [], []],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'TT100K'
}

coco = {  # ENABLE AUGMENTATIONS!!
    'num_classes': 81,
    # 'lr_steps': (280000, 360000, 400000),  # for 300
    'lr_steps': (280000, 320000, 360000),  # for 512
    # 'max_iter': 400000,  # for 300
    'max_iter': 360000,  # for 512
    # 'feature_maps': [38, 19, 10, 5, 3, 1],  # for 300 (변경시 get_detected_info 확인)
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],  # for 512 (변경시 get_detected_info 확인)
    # 'min_dim': 300,
    'min_dim': 512,  # for 512
    # 'steps': [8, 16, 32, 64, 100, 300],  # for 300
    'steps': [8, 16, 32, 64, 128, 256, 512],  # for 512
    # 'min_sizes': [21, 45, 99, 153, 207, 261],  # for 300 (0.07, 0.15, 0.9)
    # 'max_sizes': [45, 99, 153, 207, 261, 315],  # for 300 (0.07, 0.15, 0.9)
    # 'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],  # for 512 (0.07, 0.15, 0.9)
    # 'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],  # for 512 (0.07, 0.15, 0.9)
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],  # for 512 (0.04, 0.10, 0.9)
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],  # for 512 (0.04, 0.10, 0.9)
    # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # for 300 (변경시 get_detected_info 확인)
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],  # for 512 (변경시 get_detected_info 확인)
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
