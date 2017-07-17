# --------------------------------------------------------
# LSI-Faster R-CNN
# Original work Copyright (c) 2015 Microsoft
# Modified work Copyright 2017 Carlos Guindel
# Licensed under The MIT License [see LICENSE for details]
# Originally written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os
import warnings

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, four_channels=cfg.TEST.FOURCHANNELS)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None, extra_boxes=np.zeros((0,4), dtype=np.float32), dc_boxes=np.zeros((0,4), dtype=np.float32)):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    if extra_boxes.shape[0]>0:
        assert (cfg.TEST.EXTERNAL_ROIS == True), "If you want to use external proposals, \
                                    you have to set the proper configuration parameter"

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        assert (cfg.TEST.EXTERNAL_ROIS == False)
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
        if cfg.TEST.EXTERNAL_ROIS:
            net.blobs['extra_rois'].reshape(*(extra_boxes.shape))
            sc_extra_boxes, _ = _project_im_rois(extra_boxes, im_scales)
            net.blobs['dc_rois'].reshape(*(dc_boxes.shape))
            sc_dc_boxes, _ = _project_im_rois(dc_boxes, im_scales)
    else:
        assert(cfg.TEST.EXTERNAL_ROIS == False)
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
        if cfg.TEST.EXTERNAL_ROIS:
            forward_kwargs['extra_rois'] = sc_extra_boxes
            forward_kwargs['dc_rois'] = sc_dc_boxes
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.VIEWPOINTS:
        try:
            viewpoints = blobs_out['viewpoint_pred']
        except KeyError, e:
            viewpoints = blobs_out['viewpoints_pd']
        except:
            print 'Unknown error reading the viewpoint predictions.' % str(e)

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
        if cfg.VIEWPOINTS:
            viewpoints = viewpoints[inv_index, :]

    if cfg.VIEWPOINTS:
        return scores, pred_boxes, viewpoints
    else:
        return scores, pred_boxes

def vis_detections(im, class_name, dets, gt=[], thresh=0.3):

    CLASS_COLOR = ((0,0,0),
        (0,0,255), (0,128,255),(0,192,255),(255,0,0),
        (255,128,128),(0,255,255), (255,255,255), (0,0,0))

    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    plt.cla()
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -2]
        if score > thresh:
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=tuple(CLASS_COLOR[int(dets[i, -1])][chn]/255.0 for chn in xrange(3)), linewidth=3)
                )
            #plt.title('{}  {:.3f}'.format(class_name, score))

    for ix, box in enumerate(gt['boxes']):
      if gt['gt_classes'][ix] > 0:
        plt.gca().add_patch(
                plt.Rectangle((box[0], box[1]),
                              box[2] - box[0],
                              box[3] - box[1], fill=False,
                              edgecolor=(1,1,1), linewidth=2)
                )
      else:
        plt.gca().add_patch(
                plt.Rectangle((box[0], box[1]),
                              box[2] - box[0],
                              box[3] - box[1], fill=False,
                              edgecolor=(0,0,0), linewidth=2)
                )
    plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    den = np.sum(np.exp(x), axis=1)
    return np.exp(x) / den[:, np.newaxis]

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    if vis:
        from datasets.kitti import kitti
        kitti = kitti("valsplit")
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score, [cfg.VIEWP_BINS x viewpoint prob. dist])
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]


    output_dir = get_output_dir(imdb, net)

    cache_file = os.path.join(output_dir, 'detections.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
            #print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            print 'Detections cache loaded'
            warnings.warn("PLEASE MAKE SURE THAT YOU REALLY WANT TO USE THE CACHE!", UserWarning)
            #return roidb
    else:

        # timers
        _t = {'im_detect' : Timer(), 'misc' : Timer()}

        if not cfg.TEST.HAS_RPN:
            roidb = imdb.roidb
        ndetections = 0

        for i, img_file in enumerate(imdb.image_index):

            if vis:
                detts = np.empty([0, 6])

            # filter out any ground truth boxes
            if cfg.TEST.HAS_RPN:
                box_proposals = None
            else:
                # The roidb may contain ground-truth rois (for example, if the roidb
                # comes from the training or val split). We only want to evaluate
                # detection on the *non*-ground-truth rois. We select those the rois
                # that have the gt_classes field set to 0, which means there's no
                # ground truth.
                if cfg.TEST.GTPROPOSALS:
                  box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] > -1]
                else:
                  box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

            if box_proposals is not None and box_proposals.shape[0] <= 0:
              # if there are no proposals....
              scores = np.empty((0, imdb.num_classes), dtype=np.float32)
              boxes = np.empty((0, imdb.num_classes*4), dtype=np.float32)
              if cfg.VIEWPOINTS:
                  viewpoints = np.empty((0, imdb.num_classes*cfg.VIEWP_BINS), dtype=np.float32)
            else:
              if cfg.TEST.FOURCHANNELS:
                  im = cv2.imread(imdb.image_path_at(i), cv2.IMREAD_UNCHANGED)
              else:
                  im = cv2.imread(imdb.image_path_at(i))

              _t['im_detect'].tic()
              if cfg.VIEWPOINTS:
                  scores, boxes, viewpoints = im_detect(net, im, box_proposals)
              else:
                  scores, boxes = im_detect(net, im, box_proposals)
              _t['im_detect'].toc()
              print _t['im_detect'].diff

            _t['misc'].tic()
            # skip j = 0, because it's the background class
            for j in xrange(1, imdb.num_classes):
                inds = np.where(scores[:, j] > thresh)[0]
                ndetections += len(inds)
                cls_scores = scores[inds, j]
                cls_boxes = boxes[inds, j*4:(j+1)*4]
                if cfg.VIEWPOINTS:
                    # Softmax is only performed over the class N_BINSx "slot"
                    # (that is why we apply it outside Caffe)
                    cls_viewp = softmax(viewpoints[inds, j*cfg.VIEWP_BINS:(j+1)*cfg.VIEWP_BINS])
                    # Assert that the result from softmax makes sense
                    assert(all(abs(np.sum(cls_viewp, axis=1)-1)<0.1))
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis], cls_viewp)) \
                        .astype(np.float32, copy=False)
                else:
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                        .astype(np.float32, copy=False)
                if cfg.TEST.DO_NMS:
                  if cfg.USE_CUSTOM_NMS:
                      if cfg.VIEWPOINTS:
                          nms_returns = nms(cls_dets[:,:-cfg.VIEWP_BINS], cfg.TEST.NMS, force_cpu=True)
                      else:
                          nms_returns = nms(cls_dets, cfg.TEST.NMS, force_cpu=True)
                      if nms_returns:
                          keep = nms_returns[0]
                          suppress = nms_returns[1]
                      else:
                          keep = []
                  else:
                      if cfg.VIEWPOINTS:
                          keep = nms(cls_dets[:,:-cfg.VIEWP_BINS], cfg.TEST.NMS)
                      else:
                          keep = nms(cls_dets, cfg.TEST.NMS)
                  cls_dets = cls_dets[keep, :]
                else:
                  cls_dets=cls_dets[cls_dets[:,-9].argsort()[::-1],:]

                if vis:
                  pre_detts = np.hstack((np.array(cls_dets[:,:5]), j*np.ones((np.array(cls_dets[:,:5]).shape[0],1))))
                  detts = np.vstack((detts, pre_detts))

                all_boxes[j][i] = cls_dets

            if vis:
              gt_roidb = kitti._load_kitti_annotation(img_file)
              vis_detections(im, imdb.classes, detts, gt_roidb)

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                if cfg.VIEWPOINTS:
                    image_scores = np.hstack([all_boxes[j][i][:, -9]
                                                for j in xrange(1, imdb.num_classes)])
                else:
                    image_scores = np.hstack([all_boxes[j][i][:, -1]
                                                for j in xrange(1, imdb.num_classes)])

                if len(image_scores) > max_per_image:
                    # We usually don't want to do this
                    print "WARNING! Limiting the number of detections"
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in xrange(1, imdb.num_classes):
                        if cfg.VIEWPOINTS:
                            keep = np.where(all_boxes[j][i][:, -9] >= image_thresh)[0]
                        else:
                            keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]
            _t['misc'].toc()

            print 'im_detect: {:d}/{:d} - {:d} detections - {:.3f}s {:.3f}s' \
                  .format(i + 1, num_images, ndetections,_t['im_detect'].average_time,
                          _t['misc'].average_time)

        det_file = os.path.join(output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
