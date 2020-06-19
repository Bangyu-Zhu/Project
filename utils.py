import os
import torch
import shutil
import numpy as np


def iou(pred, target, num_classes=4):
    ious = []
    # background class (o) is ignored
    for cls in range(1, num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
        return np.array(ious)


def save_checkpoint(state_dict, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state_dict, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def class_color_mapping():
    class_color_map = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 0, 0)}

