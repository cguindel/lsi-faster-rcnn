# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# Modified at UC3M by cguindel
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv

DEBUG = False

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        anchor_scales = cfg.ANCHOR_SCALES
        self._anchors = generate_anchors(scales=np.array(anchor_scales), ratios=cfg.ANCHOR_ASPECT_RATIOS)
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x1, y1, x2, y2, label, viewpoint)
        gt_boxes = bottom[1].data
        # We don't want orientation here
        gt_boxes = gt_boxes[:,:-1]
        # im_info
        im_info = bottom[2].data[0, :]

        if DEBUG:
            if len(bottom)>2:
                img = bottom[3].data
            np.set_printoptions(threshold=np.nan)
            print '~~~ ANCHOR_TARGET_LAYER ~~~'
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ), dtype=np.float32)
        labels.fill(-1)

        """
        This is hardly modified from the original because we want to
        manage DontCare labels. They are assigned -1 class index.
        Later layers already knew how to handle -1.
        """
        # Find care/dontcare gt_boxes indices
        dontcare_gt_inds = np.where(gt_boxes[:,4]<0)[0]
        care_gt_inds = np.where(gt_boxes[:,4]>-1)[0]

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        # argmax_overlaps: assigned gt_boxes for each anchor
        argmax_overlaps = overlaps.argmax(axis=1)

        # max_overlaps: overlap value over the corresponding gt_box
        # for each anchor
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

        # gt_argmax_overlaps: assigned anchor for each gt_box (i.e. anchor
        # indices with greater overlapping with each gt_box
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        # gt_max_overlaps: overlap value over the corresponding max-anchor
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]

        # Find anchor indices whose greater overlapping is over a dontcare
        # gt_box
        dontcare_anchor_inds_list = [
            anchor_ind for anchor_ind, gt_box in enumerate(argmax_overlaps)
            if gt_box in dontcare_gt_inds]
        dontcare_anchor_inds = np.array(dontcare_anchor_inds_list)

        # gt_dontcare_argmax_overlaps: Anchor indices that are max overlappers
        # over a dontcare gt_box
        gt_dontcare_argmax_overlaps = np.array([gt_argmax_overlaps[i]
                                        for i in dontcare_gt_inds])
        # gt_care_argmax_overlaps: Anchor indices that are max overlappers over
        # a dontcare gt_box
        gt_care_argmax_overlaps = np.array([gt_argmax_overlaps[i]
                                    for i in care_gt_inds])

        # Overlap values for the previous anchor indices
        gt_dontcare_max_overlaps = overlaps[gt_dontcare_argmax_overlaps,
                                   dontcare_gt_inds]
        gt_care_max_overlaps = overlaps[gt_care_argmax_overlaps,
                               care_gt_inds]

        # gt_argmax_overlaps: Anchor indices that are max overlappers over
        # any gt_box
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        # Anchor indices that are max overlappers over care/dontcare gt_boxes
        gt_dontcare_argmax_overlaps = np.where(
            overlaps[:,dontcare_gt_inds] == gt_dontcare_max_overlaps)[0]
        gt_care_argmax_overlaps = np.where(
            overlaps[:,care_gt_inds] == gt_care_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # Configuration performs this by default
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_care_argmax_overlaps] = 1
        labels[gt_dontcare_argmax_overlaps] = -1

        # Overlap value for every care/dontcare anchor
        care_max_overlaps_list = [max_overlaps[anchor_ind]
                            if anchor_ind not in dontcare_anchor_inds else 0
                            for anchor_ind, overlp in enumerate(max_overlaps)]
        dontcare_max_overlaps_list = [max_overlaps[anchor_ind]
                            if anchor_ind in dontcare_anchor_inds else 0
                            for anchor_ind, overlp in enumerate(max_overlaps)]

        care_max_overlaps = np.array(care_max_overlaps_list)
        dontcare_max_overlaps = np.array(dontcare_max_overlaps_list)

        # fg label: above threshold IOU
        labels[care_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        # dontcare label: above DONTCARE threshold IoU
        labels[dontcare_max_overlaps > cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1

        # Here we will double check dontcares according to its threshold
        labels[np.where(overlaps[:,dontcare_gt_inds]>cfg.TRAIN.RPN_DONTCARE_OVERLAP)[0]] = -1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # Not active by default
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            print 'rpn: max max_overlap', np.max(max_overlaps)
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

            _vis_whats_happening(img, anchors, bbox_targets[inds_inside], labels[inds_inside])

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _vis_whats_happening(im_blob, bboxes, targets, labels):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    import random
    im = im_blob[0, :, :, :].transpose((1, 2, 0)).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, (2, 1, 0)]
    im = im.astype(np.uint8)
    plt.figure("RPN anchors no-targets")
    plt.imshow(im)
    plt.figure("RPN anchors targets")
    plt.imshow(im)
    rois = bbox_transform_inv(bboxes, targets)
    for i in xrange(bboxes.shape[0]):
        color = np.random.rand(3,1)

        if labels[i]>0:
          roi = bboxes[i,:]
          plt.figure("RPN anchors no-targets")
          plt.gca().add_patch(
              plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor=color, linewidth=4)
              )
          roi = rois[i, :]
          plt.figure("RPN anchors targets")
          plt.gca().add_patch(
              plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor=color, linewidth=4)
              )
        if labels[i]==0:
          roi = bboxes[i,:]
          plt.figure("RPN anchors no-targets")
          plt.gca().add_patch(
              plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor=color, linewidth=1)
              )
          #roi = rois[i, :]
          plt.figure("RPN anchors targets")
          plt.gca().add_patch(
              plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor=color, linewidth=1)
              )
    plt.show(block=False)
