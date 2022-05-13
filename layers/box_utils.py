# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.get_detected_info import get_anchor_nums, get_anchor_box_size

prior_info = 0


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # xmin, ymin
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2],  # cx, cy
                      boxes[:, 2:]), 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def aligned_matching(overlap, truths, threshold, cfg, fix_ratio=False, fix_ignored=False, only_1a=False):
    prior_overlaps, prior_idx = overlap.sort(1, descending=True)
    prior_overlaps_cpu_list = prior_overlaps.cpu().numpy()
    prior_idx_cpu_list = prior_idx.cpu().numpy()
    truths_cpu_list = truths.cpu().numpy()

    best_prior_overlap = []
    best_prior_idx = []

    for i in range(prior_overlaps_cpu_list.shape[0]):
        prior_overlaps_cpu = prior_overlaps_cpu_list[i].copy()
        prior_idx_cpu = prior_idx_cpu_list[i].copy()
        truths_cpu = truths_cpu_list[i]

        # select default box with maximum IoU.
        max_prior_overlap = prior_overlaps_cpu[0]
        allowed_priors = []
        for i, prior_overlap in enumerate(prior_overlaps_cpu):
            if max_prior_overlap > prior_overlap + threshold:
                if not prior_overlap == -1:
                    break
            else:
                allowed_priors.append(i)
        prior_overlaps_cpu = prior_overlaps_cpu[allowed_priors]
        prior_idx_cpu = prior_idx_cpu[allowed_priors]

        # select default box with similar ratio with ground truth.
        allowed_priors = []
        truth_ratio = (truths_cpu[2] - truths_cpu[0]) / (truths_cpu[3] - truths_cpu[1])
        if fix_ratio:
            ratio_seperator = [np.sqrt(2), 1/np.sqrt(2), np.sqrt(3)*np.sqrt(2), 1/np.sqrt(3)*np.sqrt(2)]
            if truth_ratio > ratio_seperator[0]:
                if truth_ratio > ratio_seperator[2]:
                    gt_ratios = 4
                else:
                    gt_ratios = 2
            elif truth_ratio < ratio_seperator[1]:
                if truth_ratio < ratio_seperator[3]:
                    gt_ratios = 5
                else:
                    gt_ratios = 3
            else:
                gt_ratios = 0
        elif only_1a:
            gt_ratios = 0
        else:
            ratio_list = [1, -100, 2, 0.5, 3, 1 / 3]
            gt_ratios = np.abs(np.array(ratio_list) - truth_ratio).argmin()
        for i, prior_id in enumerate(prior_idx_cpu):
            prior_ratio = get_anchor_box_size(prior_id, cfg['feature_maps'], cfg['aspect_ratios'])
            if fix_ratio:
                if prior_ratio == 1:  # append big 1:1 default box.
                    prior_ratio = 0
            if prior_ratio == gt_ratios:
                allowed_priors.append(i)
            else:
                # get types of default box ratio of each feature map.
                anchor_size_list, accumulated_slice_list = get_anchor_nums(cfg['feature_maps'], cfg['aspect_ratios'])
                for j, accumulated_slice in enumerate(accumulated_slice_list):
                    if prior_id < accumulated_slice:
                        break
                # if ratio is out of range (ex. feature map with no 1:3, 3:1 default box)
                if (anchor_size_list[j] - 1) < gt_ratios:
                    # match 2:1 default box with 3:1 ground truth and so on...
                    if get_anchor_box_size(prior_id, cfg['feature_maps'], cfg['aspect_ratios']) % 2 == gt_ratios % 2:
                        allowed_priors.append(i)

        if allowed_priors:
            prior_overlaps_cpu_temp = []
            prior_idx_cpu_temp = []
            for allowed_prior in allowed_priors:
                prior_overlaps_cpu_temp.append(prior_overlaps_cpu[allowed_prior])
                prior_idx_cpu_temp.append(prior_idx_cpu[allowed_prior])
            prior_overlaps_cpu = prior_overlaps_cpu_temp
            prior_idx_cpu = prior_idx_cpu_temp

        # select default box which nearest with ground truth.
        center_sizes = prior_sizes[prior_idx_cpu]
        truth_cx = (truths_cpu[2] + truths_cpu[0]) / 2
        truth_cy = (truths_cpu[3] + truths_cpu[1]) / 2
        center_dist = np.sqrt(np.power(center_sizes[:, 0] - truth_cx, 2) + np.power(center_sizes[:, 1] - truth_cy, 2))
        idx = np.argmin(center_dist)
        best_prior_overlap.append([prior_overlaps_cpu[idx]])
        best_prior_idx.append([prior_idx_cpu[idx]])

        if fix_ignored:
            for j, prior_idx_on_one_gt in enumerate(prior_idx_cpu_list):
                for k, value_idx in enumerate(prior_idx_on_one_gt):
                    if value_idx == prior_idx_cpu[idx]:
                        prior_overlaps_cpu_list[j, k] = -1
                        break

    best_prior_overlap = torch.tensor(best_prior_overlap)
    best_prior_idx = torch.tensor(best_prior_idx)
    return best_prior_overlap, best_prior_idx


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx, matching, cfg, fix_loss=True):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when matching boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
        matching: (string) Method to match. 'legacy' or 'aligned'
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    global prior_sizes
    global prior_info

    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    if prior_info == 0:  # at first execution of match
        prior_sizes = center_size(priors).cpu().numpy()
        prior_info = []
        for i in range(len(priors)):
            prior_info.append(get_anchor_box_size(i, cfg['feature_maps'], cfg['aspect_ratios']))

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    if matching == 'legacy':
        best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    elif matching == 'aligned':
        # use aligned matching
        best_prior_overlap, best_prior_idx = aligned_matching(overlaps, truths, 1e-6, cfg)
    elif matching == 'aligned_2':
        # use fixed aligned matching
        best_prior_overlap, best_prior_idx = aligned_matching(overlaps, truths, 1e-6, cfg, fix_ratio=True)
    elif matching == 'aligned_1a':
        # use fixed aligned matching
        best_prior_overlap, best_prior_idx = aligned_matching(overlaps, truths, 1e-6, cfg, fix_ratio=False, fix_ignored=False, only_1a=True)
    elif matching == 'aligned_3':
        best_prior_overlap, best_prior_idx = aligned_matching(overlaps, truths, 1e-6, cfg, fix_ratio=True, fix_ignored=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    # Start visualization code

    import cv2

    img = np.zeros((1000, 1000, 3), np.uint8)
    obj_idx = 0
    for prior_idx in point_form(priors)[best_prior_idx]:
        prior_idx = prior_idx[0].cpu().numpy()
        # print(
        #     f'matched ratio : {np.around(((prior_idx[2] - prior_idx[0]) / (prior_idx[3] - prior_idx[1])), decimals=2)}')
        # print(int(best_prior_idx[obj_idx]))
        # print(int(obj_idx))
        # print(f"algorithm result : {get_anchor_box_size(int(best_prior_idx[obj_idx]), cfg['feature_maps'], cfg['aspect_ratios'])}")
        obj_idx += 1
        # print((prior_idx[0] * 1000, prior_idx[1] * 1000))
        # print((prior_idx[2] * 1000, prior_idx[3] * 1000))
        img = cv2.rectangle(img, (int(prior_idx[0] * 1000), int(prior_idx[1] * 1000)),
                            (int(prior_idx[2] * 1000), int(prior_idx[3] * 1000)), (0, 0, 255), 3)
    # print('truth =')
    for truth in truths:
        # print((truth[0] * 1000, truth[1] * 1000))
        # print((truth[2] * 1000, truth[3] * 1000))
        # print(f'width = {truth[2] * 1000 - truth[0] * 1000}')
        # print(f'height = {truth[3] * 1000 - truth[1] * 1000}')
        img = cv2.rectangle(img, (int(truth[0] * 1000), int(truth[1] * 1000)),
                            (int(truth[2] * 1000), int(truth[3] * 1000)), (0, 255, 0), 3)

    cv2.imwrite(
        f'./matching_output/aligned3_0.10/{int(truth[0] * 1000)},{int(truth[1] * 1000)},'
        f'{int(truth[2] * 1000)},{int(truth[3] * 1000)}.png', img)  # 이미지 저장
    # cv2.imshow('bbox', img)
    # cv2.waitKey(2000)

    # End visualization code
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior

    if fix_loss:
        # set matched box coordinate to prior box data.
        matches = point_form(priors)
        # for each ground truth,
        for j in range(best_prior_idx.size(0)):
            # ensure every gt matches with its prior of max overlap
            best_truth_idx[best_prior_idx[j]] = j
            # write ground truth data on matched default box.
            matches[best_prior_idx[j]] = truths[j]
            # write ground truth data on default box which has iou over threshold.
            multi_matching_idx = torch.where(best_truth_overlap >= threshold)
            for k in multi_matching_idx[0]:
                matches[k] = truths[best_truth_idx[k]]
    else:
        for j in range(best_prior_idx.size(0)):
            # ensure every gt matches with its prior of max overlap
            best_truth_idx[best_prior_idx[j]] = j
        matches = truths[best_truth_idx]  # Shape: [num_priors,4]

    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
