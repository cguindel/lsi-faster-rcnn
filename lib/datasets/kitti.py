# --------------------------------------------------------
# LSI-Faster R-CNN
# Original work Copyright (c) 2015 Microsoft
# Modified work Copyright 2018 Carlos Guindel
# Licensed under The MIT License [see LICENSE for details]
# Based on code written by Ross Girshick
# --------------------------------------------------------
# Note:
# KITTI Object Dataset is expected to be at $FRCN_ROOT/data/kitti/images
# --------------------------------------------------------

from datasets.imdb import imdb
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import uuid
from fast_rcnn.config import cfg
import math
import matplotlib.pyplot as plt
from utils import angles

DEBUG = False
STATS = False   # Retrieve stats from the KITTI dataset

class kitti(imdb):
    def __init__(self, image_set, devkit_path=None):
        imdb.__init__(self, 'kitti_' + image_set)
        self._image_set = image_set # Custom image split

        # Paths
        self._devkit_path = os.path.join(cfg.DATA_DIR, 'kitti') if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, cfg.TRAIN.KITTI_FOLDER)

        self._kitti_set = 'testing' if image_set[:4]=='test' else 'training' # training / testing

        self._image_ext = '.png'

        self._classes = tuple(cfg.CLASSES)
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.dollar_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        if STATS:
            self.aspect_ratios = []
            self.widths = []
            self.heights = []
            self.areas = []

        assert os.path.exists(self._devkit_path), \
                'KITTI path does not exist at data: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, self._kitti_set, 'image_2',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)

        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /KITTI/object/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._devkit_path, 'lists',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().zfill(6) for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file) and cfg.TRAIN.KITTI_USE_CACHE:
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]
        if STATS:
          plt.figure("Ratios")
          plt.hist(self.aspect_ratios, bins=[0, 0.15, 0.25, 0.35, 0.45, 0.55,
                                            0.65, 0.75, 0.85, 0.95, 1.05, 1.15,
                                            1.25, 1.35, 1.45, 1.55, 1.65, 1.75,
                                            1.85, 1.95, 2.05, 2.15, 2.25, 2.35,
                                            2.45, 2.55, 2.65, 2.75, 2.85, 2.95,
                                            3.05, 3.15, 3.25, 3.35, 3.45, 3.55,
                                            3.65, 3.75, 3.85, 3.95, 4.05], normed=True)
          plt.show()
          plt.figure("Widths")
          plt.hist(self.widths)
          plt.show()
          plt.figure("Heights")
          plt.hist(self.heights)
          plt.show()
          plt.figure("Areas")
          plt.hist(self.areas, bins=[50, 150, 250, 350, 450, 550, 650, 750, 850,
                                    950, 1050, 1150, 1250, 1350, 1450, 1550, 1650,
                                    1750, 1850, 1950, 2050, 2150, 2250, 2350, 2450,
                                    2550, 2650, 2750, 2850, 2950, 3050, 3150, 3250,
                                    3350, 3450, 3550])
          plt.show()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, self._kitti_set, 'label_2',
                                index + '.txt')
        print 'Loading: {}'.format(filename)

        pre_objs = np.genfromtxt(filename, delimiter=' ',
               names=['type', 'truncated', 'occluded', 'alpha',
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                        'dimensions_1', 'dimensions_2', 'dimensions_3',
                        'location_1', 'location_2', 'location_3',
                        'rotation_y', 'score'], dtype=None)

        # Just in case no objects are present
        if (pre_objs.ndim < 1):
            pre_objs = np.array(pre_objs, ndmin=1)

        boxes = np.empty((0, 4), dtype=np.float32)
        gt_classes = np.empty((0), dtype=np.int32)
        overlaps = np.empty((0, self.num_classes), dtype=np.float32)
        gt_orientation = np.empty((0), dtype=np.float32)
        seg_areas = np.empty((0), dtype=np.float32)

        external_rois = np.empty((0, 4), dtype=np.float32)
        dc_rois = np.empty((0, 4), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        saved = 0
        for obj in pre_objs:
            x1 = obj['bbox_xmin']
            y1 = obj['bbox_ymin']
            x2 = obj['bbox_xmax']
            y2 = obj['bbox_ymax']

            if cfg.PREFILTER:
                if  x1<0 or x1>cfg.PREFILTER_WIDTH or \
                    y1<0 or y1>cfg.PREFILTER_HEIGHT or \
                    x2<0 or x2>cfg.PREFILTER_WIDTH or \
                    y2<0 or y2>cfg.PREFILTER_HEIGHT:
                    continue

            # Easy / medium / hard restraints
            if obj['type'] == 'UNDEFINED':
                external_rois = np.vstack((external_rois, [x1, y1, x2, y2]))
                continue
            elif obj['type'] == 'DontUse':
                dc_rois = np.vstack((dc_rois, [x1, y1, x2, y2]))
                continue # Jorge's mini squares separately used

            # Remap sibling classes
            cls_name = obj['type'].strip()
            if cls_name in cfg.CLASSES_MAP[0]:
                cls_name = cfg.CLASSES_MAP[0][cls_name]

            if cls_name not in self._classes \
            or (obj['truncated']>cfg.MAX_TRUNCATED) \
            or (obj['occluded']>cfg.MAX_OCCLUDED) \
            or (y2-y1<cfg.MIN_HEIGHT) \
            or (x1<cfg.MIN_X1):
                gt_classes = np.hstack((gt_classes,-1))
            else:
                cls = self._class_to_ind[str(cls_name)]
                gt_classes = np.hstack((gt_classes,cls))

                overlap_row = np.zeros(self.num_classes)
                overlap_row[cls] = 1.0
                overlaps = np.vstack((overlaps,overlap_row))

                if STATS:
                  self.aspect_ratios.append((y2 - y1)/(x2 - x1))
                  self.widths.append((x2 - x1))
                  self.heights.append((y2 - y1))
                  self.areas.append((x2 - x1)*(y2 - y1))
            boxes = np.vstack((boxes, [x1, y1, x2, y2]))
            seg_areas = np.hstack((seg_areas, (x2 - x1 + 1) * (y2 - y1 + 1)))
            gt_orientation = np.hstack((gt_orientation, obj['alpha']))

            # Undefined angle if class is not valid
            if gt_classes[-1] == -1:
                gt_orientation[-1] = -10
            else:
                assert gt_orientation[-1] < cfg.VIEWP_BINS

        overlaps = scipy.sparse.csr_matrix(overlaps)

        if DEBUG:
            print index
            print {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'gt_viewpoints' : gt_orientation,
                    'flipped' : False,
                    'seg_areas' : seg_areas
                    }
            print 'overlaps', overlaps.todense()

        if cfg.VIEWPOINTS:
            if not cfg.TRAIN.EXTERNAL_ROIS:
                return {'boxes' : boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps' : overlaps,
                        'gt_viewpoints' : gt_orientation,
                        'flipped' : False,
                        'seg_areas' : seg_areas
                        }
            else:
                return {'boxes' : boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps' : overlaps,
                        'gt_viewpoints' : gt_orientation,
                        'flipped' : False,
                        'seg_areas' : seg_areas,
                        'external_rois': external_rois,
                        'dc_rois': dc_rois
                        }
        else:
            return {'boxes' : boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps' : overlaps,
                    'flipped' : False,
                    'seg_areas' : seg_areas
                    }

    def dollar_roidb(self):
        """
        Return the database of Dollar Edges regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        print 'Called dollar_roidb'
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_dollar_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_dollar_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_dollar_roidb(self, gt_roidb):
        print 'Called _load_dollar_roidb'
        filename = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'edges_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Dollar boxes data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def evaluate_detections(self, all_boxes, output_dir):
        """
        Detections are written in a file with KITTI object database format
        """

        comp_id = '{}'.format(os.getpid())

        path = os.path.join(output_dir, comp_id, '')

        for im_ind, index in enumerate(self.image_index):
            print 'Writing {} KITTI results file'.format(index)
            filename = path + index + '.txt'
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    else:
                        write_cls = cls
                    dets = all_boxes[cls_ind][im_ind]

                    if dets == []:
                        continue

                    for k in xrange(dets.shape[0]):
                        if cfg.VIEWPOINTS:
                            if cfg.CONTINUOUS_ANGLE or cfg.SMOOTH_L1_ANGLE:
                                estimated_angle = dets[k, -1]
                                estimated_score = math.log(dets[k, -2])
                            elif cfg.KL_ANGLE or cfg.TEST.KL_ANGLE:
                                estimated_angle = angles.kl_angle(dets[k, -cfg.VIEWP_BINS:], cfg.VIEWP_BINS, cfg.VIEWP_OFFSET)
                                estimated_score = math.log(dets[k, -cfg.VIEWP_BINS-1])
                            else:
                                probs = dets[k, -cfg.VIEWP_BINS:]
                                max_bin = np.argmax(probs)
                                assert max_bin < cfg.VIEWP_BINS
                                if cfg.TEST.W_ALPHA:
                                    estimated_angle = angles.walpha_angle(probs, cfg.VIEWP_BINS, cfg.VIEWP_OFFSET)
                                else:
                                    estimated_angle = angles.bin_center_angle(probs, cfg.VIEWP_BINS, cfg.VIEWP_OFFSET)

                                # log(score) to avoid score 0.0
                                estimated_score = math.log(dets[k, -cfg.VIEWP_BINS-1])
                        else:
                            estimated_angle = -10
                            estimated_score = math.log(dets[k, -1])

                        # KITTI expects 0-based indices
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.6f}\n'.
                                format(write_cls,
                                       estimated_angle,
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3],
                                       estimated_score))

        print 'Results were saved in', filename

    def competition_mode(self, on):
        """
        Not implemented
        """
        print 'Wow, competition mode. Doing nothing...'

if __name__ == '__main__':
    d = kitti('training')
    res = d.roidb
    from IPython import embed; embed()
