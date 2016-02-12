# KITTI Object Dataset is expected to be at $FRCN_ROOT/data/kitti
# image_set
# training:kitti/object/testing/image_2/[from 000 to 007480.png]
# testing: 007517.png

import datasets
import datasets.kitti
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import math

class kitti(datasets.imdb):
    def __init__(self, image_set, devkit_path=None):
        datasets.imdb.__init__(self, 'kitti_' + image_set)
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'object')
        self._classes = ('__background__', # always index 0
                         'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.dollar_roidb
        # 8 orientation bins:
        self._orientations = (0.39, 1.18, 1.96, 2.75,
                                -2.75, -1.96, -1.18, -0.39, 1.57)

        assert os.path.exists(self._devkit_path), \
                'KITTI devkit path does not exist: {}'.format(self._devkit_path)
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
        image_path = os.path.join(self._data_path, self._image_set, 'image_2',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)        #bool_arr = np.array([p_obj['type'].strip() != 'DontCare' for p_obj in pre_objs])

        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /KITTI/object/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]

        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'kitti')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, self._image_set, 'label_2',
                                index + '.txt')
        print 'Loading: {}'.format(filename)

        pre_objs = np.genfromtxt(filename, delimiter=' ',
               names=['type', 'truncated', 'occluded', 'alpha',
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                        'dimensions_1', 'dimensions_2', 'dimensions_3',
                        'location_1', 'location_2', 'location_3',
                        'rotation_y', 'score'], dtype=None)

        # Just in case there are no objects
        if (pre_objs.ndim < 1):
            pre_objs = np.array(pre_objs, ndmin=1)

        num_objs = pre_objs.size

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        gt_orientation = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(pre_objs):
            x1 = obj['bbox_xmin']
            y1 = obj['bbox_ymin']
            x2 = obj['bbox_xmax']
            y2 = obj['bbox_ymax']
            if obj['type'].strip() != 'DontCare' :
                cls = self._class_to_ind[str(obj['type'].strip())]
                gt_classes[ix] = cls
            else :
                #DontCare is assigned index -1
                gt_classes[ix] = -1
            # KITTI boxes have one decimal place
            boxes[ix, :] = [round(x1), round(y1), round(x2), round(y2)]
            overlaps[ix, cls] = 1.0
            if obj['alpha'] != -10 :
                # Assign an orientation bin for every object
                if obj['alpha']<0 :
                    angle = math.floor((6.28319+obj['alpha'])/0.785398)
                else:
                    angle = math.floor((obj['alpha'])/0.785398)
            else:
                angle = -10;

            gt_orientation[ix] = angle
            # Checks
            if gt_classes[ix] == -1:
                assert gt_orientation[ix]==-10
            else:
                assert gt_orientation[ix]<8

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'gt_orientations' : gt_orientation}

    def dollar_roidb(self):
        """
        Return the database of Dollar Edges regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_dollar_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_dollar_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_dollar_roidb(self, gt_roidb):
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

        path = os.path.join(self._devkit_path, 'results', comp_id, '')

        for im_ind, index in enumerate(self.image_index):
            print 'Writing {} VOC results file'.format(index)
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
                        angle = dets[k, -9:-1]
                        assert np.amax(angle) < 8
                        angle_bin = np.argmax(angle)
                        # KITTI expects 0-based indices
                        f.write('{:s} -1 -1 {:.2f} {:.1f} {:.1f} {:.1f} {:.1f} -1 -1 -1 -1000 -1000 -1000 -10 {:.3f}\n'.
                                format(write_cls,
                                       self._orientations[angle_bin],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3],
                                       dets[k, -9]))

    def competition_mode(self, on):
        """
        Not implemented
        """
        print 'Wow, competition mode. Doing nothing...'

if __name__ == '__main__':
    d = datasets.kitti('training')
    res = d.roidb
    from IPython import embed; embed()
