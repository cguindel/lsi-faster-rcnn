# --------------------------------------------------------
# LSI-Faster R-CNN
# Original work Copyright (c) 2015 Microsoft
# Modified work Copyright 2018 Carlos Guindel
# Licensed under The MIT License [see LICENSE for details]
# Originally written by Ross Girshick and Sean Bell
# --------------------------------------------------------

from fast_rcnn.config import cfg
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms, cpu_soft_nms
import numpy as np

# Soft NMS. +info:
#   N. Bodla, B. Singh, R. Chellappa, and L. S. Davis,
#   "Soft-NMS - Improving Object Detection With One Line of Code,"
#   in Proc. IEEE International Conference on Computer Vision (ICCV), 2017,
#   pp. 5562-5570.
def soft_nms(dets, sigma=0.5, Nt=0.3, threshold=0.001, method=1):

    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep

# Original NMS implementation
def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)
