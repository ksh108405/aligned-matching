from data.config import voc, tt100k, coco
anchor_result = [0, 0, 0, 0, 0, 0]
feature_result = [0, 0, 0, 0, 0, 0, 0]

cfg = tt100k

def get_anchor_nums(feature_map=cfg['feature_maps'], anchor_box=cfg['aspect_ratios'], square_anchor_num=2):
    accumulated_slice_list = []
    slice_list = []
    anchor_size_list = []
    before = 0
    # square_anchor_num은 정사각형 앵커박스 개수 (0 or 1 or 2)
    for feature, anchor in zip(feature_map, anchor_box):
        # print(f'f : {feature}, a : {anchor}')
        # anchor_size = 1:1, big 1:1, 2:1, 1:2, 1:3, 3:1 등등 앵커박스 총 갯수
        anchor_size_list.append(len(anchor) * 2 + square_anchor_num)
        # slice = 추론에 사용된 피쳐맵 넓이 * 앵커박스 총 갯수
        slice_list.append(feature * feature * anchor_size_list[-1])
        # accumulated_slice = 지금까지의 앵커박스의 총 갯수의 합
        accumulated_slice_list.append(before + slice_list[-1])
        before = accumulated_slice_list[-1]
    return anchor_size_list, accumulated_slice_list


def get_anchor_box_size(idx, feature_map=cfg['feature_maps'], anchor_box=cfg['aspect_ratios'], square_anchor_num=2):
    anchor_size_list, accumulated_slice_list = get_anchor_nums(feature_map, anchor_box, square_anchor_num)
    # print(anchor_size_list, accumulated_slice_list)
    for i, elem in enumerate(accumulated_slice_list):
        if idx <= elem:
            # print(idx)
            # print(idx - (0 if i == 0 else accumulated_slice_list[i - 1]))
            # n = ((idx - (0 if i == 0 else accumulated_slice_list[i - 1])) // tt100k['num_classes']) % anchor_size_list[i]
            n = (idx - (0 if i == 0 else accumulated_slice_list[i - 1])) % anchor_size_list[i]
            return int(n)
            # 만일 accumulated_slice의 원소 elem이 함수 패러미터로 전달된 idx보다 크거나 같으면,
            # (idx - 그 이전까지의 피쳐맵의 앵커박스 갯수의 합) - 2 에, 그 피쳐맵의 1셀당 앵커박스 갯수를 나눈 나머지를 n이라 한다.
            # index가 1부터 시작하기 위해 n+1을 반환한다.
            # idx는 conf layer의 weight에서 가장 확률이 높은 결과의 idx이다.
            # 이때, shape =
            # (1번 피쳐맵 면적 * 1번 피쳐맵 1셀당 앵커박스 갯수 * (클래스 갯수 + 1)) +
            # (2번 피쳐맵 면적 * 2번 피쳐맵 1셀당 앵커박스 갯수 * (클래스 갯수 + 1)) + ...
            # n = ((idx - (0 if i == 0 else accumulated_slice_list[i-1]))-2) % anchor_size_list[i]
            # n = (idx - (0 if i == 0 else accumulated_slice_list[i - 1])) % anchor_size_list[i]