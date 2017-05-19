#!/usr/bin/env python

# --------------------------------------------------------
# LSI-Faster R-CNN
# Copyright (c) 2017 Carlos Guindel
# Licensed under The MIT License [see LICENSE for details]
# Developed at Universidad Carlos III de Madrid
# --------------------------------------------------------

"""
Demo script showing detections and viewpoint inference in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', # always index 0
        'Car', 'Van', 'Truck', 'Pedestrian',
        'Person_sitting', 'Cyclist', 'Tram')

CLASS_COLOR = ((0,0,0),
        (0,0,255), (0,128,255),(0,192,255),(255,255,0),
        (128,128,0),(0,255,255), (255,255,255), (0,0,0))

NETS = {'vgg16': ('VGG16',
                  'lsi_vgg16.caffemodel',
                  'lsi_models',
                  'leaderboard')}

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    den = np.sum(np.exp(x), axis=1)
    return np.exp(x) / den[:, np.newaxis]

def draw_detections(image, scores, boxes, viewpoints, thresh=0.5):

    # Visualize detections for each class
    alpha = np.zeros_like(image)
    nice = np.zeros_like(image)

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        cls_viewp = softmax(viewpoints[:, \
            cls_ind*cfg.VIEWP_BINS:(cls_ind+1)*cfg.VIEWP_BINS])

        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis],
                          cls_viewp)).astype(np.float32)

        keep = nms(dets[:,:-cfg.VIEWP_BINS], NMS_THRESH)
        dets = dets[keep, :]

        for i, det in enumerate(dets):

            bbox = det[:4]
            score = det[4]
            width = int(bbox[2]) - int(bbox[0])
            height = int(bbox[3]) - int(bbox[1])

            if score>thresh:

                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                angle_bin = np.argmax(det[-cfg.VIEWP_BINS:])

                cv2.rectangle(image,(int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])), CLASS_COLOR[cls_ind], 2)

                cv2.rectangle(alpha,(int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),CLASS_COLOR[cls_ind], -1)

                if angle_bin > 3:
                    start_arrow = (int(bbox[0])+width/2, int(bbox[1])-5)
                    if angle_bin == 4:
                        end_arrow = (int(bbox[0])+width/2-48, int(bbox[1])-15)
                    elif angle_bin == 5:
                        end_arrow = (int(bbox[0])+width/2-8, int(bbox[1])-25)
                    elif angle_bin == 6:
                        end_arrow = (int(bbox[0])+width/2+8, int(bbox[1])-25)
                    elif angle_bin == 7:
                        end_arrow = (int(bbox[0])+width/2+48, int(bbox[1])-15)

                    cv2.putText(image, '{:s} ({:.0f}%)'.format(cls, score*100),
                        (int(bbox[0]), int(bbox[3])+15), cv2.FONT_HERSHEY_DUPLEX,
                        0.4, CLASS_COLOR[cls_ind])
                else:
                    start_arrow = (int(bbox[0])+width/2, int(bbox[3])+5)
                    if angle_bin == 0:
                        end_arrow = (int(bbox[0])+width/2+48, int(bbox[3])+15)
                    elif angle_bin == 1:
                        end_arrow = (int(bbox[0])+width/2+8, int(bbox[3])+25)
                    elif angle_bin == 2:
                        end_arrow = (int(bbox[0])+width/2-8, int(bbox[3])+25)
                    elif angle_bin == 3:
                        end_arrow = (int(bbox[0])+width/2-48, int(bbox[3])+15)

                    cv2.putText(image, '{:s} ({:.0f}%)'.format(cls, score*100),
                        (int(bbox[0]), int(bbox[1])-10), cv2.FONT_HERSHEY_DUPLEX,
                        0.4, CLASS_COLOR[cls_ind])

                cv2.arrowedLine(image, start_arrow, end_arrow, \
                    CLASS_COLOR[cls_ind], 3, cv2.LINE_AA, 0, 0.6)

    nice = cv2.addWeighted(alpha, 0.3, image, 1, 0)

    cv2.imshow("demo", nice)
    cv2.waitKey(0)

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, \
        'kitti', 'images', 'testing', 'image_2', image_name)
    print im_file
    im = cv2.imread(im_file)


    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, viewpoints = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    draw_detections(im, scores, boxes, viewpoints)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.VIEWPOINTS = True
    cfg.MODELS_DATASET = 'kitti'
    cfg.MODELS_DIR = os.path.abspath(os.path.join(cfg.ROOT_DIR, 'models', \
        cfg.MODELS_DATASET))

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            NETS[args.demo_net][3], 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, NETS[args.demo_net][2],
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_lsi_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _= im_detect(net, im)

    im_names = ['000001.png', '000002.png', '000005.png',
                '000008.png', '000012.png', '000014.png'
    ]

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(im_name)
        demo(net, im_name)
