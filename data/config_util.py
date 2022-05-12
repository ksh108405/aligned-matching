import math


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


size_index = {'300': 0, '512': 1, '1024': 2}
conv4_3_size = {'0.10': [10, 20, 90], '0.07': [7, 15, 90], '0.04': [4, 10, 90], '0.02': [2, 5, 90], '0.01': [1, 3, 90]}
