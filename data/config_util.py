import math
import numpy as np


def get_min_max_size(network_size, scale_list):
    conv4_3_scale = scale_list[0]
    min_scale = scale_list[1]
    max_scale = scale_list[2]
    source_num = {'300': 6, '512': 7, '1024': 8}[str(network_size)]
    scales = [conv4_3_scale * 0.01]
    for i in range(1, source_num + 1):
        scales.append(
            (min_scale + math.floor((max_scale - min_scale) / (source_num - 2)) * (i - 1)) * 0.01)
    min_sizes = [round(x * network_size, 2) for x in scales[:-1]]
    max_sizes = [round(x * network_size, 2) for x in scales[1:]]
    return [min_sizes, max_sizes]


def get_kmeans_size(network_size, dataset):
    if dataset == 'TT100K':
        if network_size == 300:
            size_array = [[0.0091, 0.0103],
                          [0.0155, 0.0180],
                          [0.0246, 0.0277],
                          [0.0340, 0.0413],
                          [0.0498, 0.0589],
                          [0.0800, 0.0882]]
        elif network_size == 512:
            size_array = [[0.0090, 0.0102],
                          [0.0152, 0.0177],
                          [0.0238, 0.0269],
                          [0.0349, 0.0388],
                          [0.0279, 0.0594],
                          [0.0539, 0.0568],
                          [0.0799, 0.0904]]
        elif network_size == 1024:
            size_array = [[0.0083, 0.0094],
                          [0.0130, 0.0148],
                          [0.0190, 0.0220],
                          [0.0277, 0.0312],
                          [0.0268, 0.0592],
                          [0.0392, 0.0429],
                          [0.0561, 0.0595],
                          [0.0805, 0.0907]]
        else:
            raise NotImplementedError
    elif dataset == 'VOC':
        if network_size == 300:
            size_array = [[0.0847, 0.1232],
                          [0.2199, 0.3035],
                          [0.2764, 0.5609],
                          [0.7150, 0.4492],
                          [0.4588, 0.8204],
                          [0.8752, 0.8555]]
        elif network_size == 512:
            size_array = [[0.0863, 0.1187],
                          [0.1865, 0.3098],
                          [0.2474, 0.6007],
                          [0.4809, 0.3937],
                          [0.4807, 0.8193],
                          [0.8378, 0.5094],
                          [0.8771, 0.8854]]
        elif network_size == 1024:
            size_array = [[0.0774, 0.1095],
                          [0.1802, 0.2622],
                          [0.2133, 0.4876],
                          [0.4853, 0.4016],
                          [0.3196, 0.7508],
                          [0.8518, 0.4970],
                          [0.5956, 0.8195],
                          [0.9063, 0.8890]]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return size_array


size_index = {'300': 0, '512': 1, '1024': 2}
conv4_3_size = {'0.1': [10, 20, 90], '0.07': [7, 15, 90], '0.04': [4, 10, 90],
                '0.03': [3, 8, 90], '0.02': [2, 5, 90], '0.01': [1, 3, 90]}
