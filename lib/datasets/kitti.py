# KITTI Object Dataset is expected to be at [PRE_PATH]/fast-rcnn/kitti[/object]
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
        self._orientations = (0.39, 1.18, 1.96, 2.75, -2.75, -1.96, -1.18, -0.39, 1.57)

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
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]

        # print 'Last index --> {:d}'.format(len(image_index))
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
        filename = os.path.join(self._data_path, self._image_set, 'label_2', index + '.txt')
        print 'Loading: {}'.format(filename)

        pre_objs = np.genfromtxt(filename, delimiter=' ',
               names=['type', 'truncated', 'occluded', 'alpha',
                        'bbox_xmin', 'bbox_ymin', 'bbox_xmax', 'bbox_ymax',
                        'dimensions_1', 'dimensions_2', 'dimensions_3',
                        'location_1', 'location_2', 'location_3',
                        'rotation_y', 'score'], dtype=None)

        if (pre_objs.ndim < 1):
            pre_objs = np.array(pre_objs, ndmin=1)

        #bool_arr = np.array([p_obj['type'].strip() != 'DontCare' for p_obj in pre_objs])
        #objs = pre_objs[bool_arr];

        #if (objs.ndim < 1):
        #    objs = np.array(objs, ndmin=1)

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
            #cls = self._class_to_ind[
            #         str(obj['type'].lower().strip() if obj['type'].strip() is not 'DontCare' else '__background__')]
            if obj['type'].strip() != 'DontCare' :
                cls = self._class_to_ind[str(obj['type'].strip())]
                gt_classes[ix] = cls
            else :
                gt_classes[ix] = -1
            boxes[ix, :] = [round(x1), round(y1), round(x2), round(y2)]
            overlaps[ix, cls] = 1.0
            if obj['alpha'] != -10 :
                if obj['alpha']<0 :
                    angle = math.floor((6.28319+obj['alpha'])/0.785398)
                else:
                    angle = math.floor((obj['alpha'])/0.785398)
            else:
                angle = -10;

            gt_orientation[ix] = angle
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
        Return the database of selective search regions of interest.
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

        comp_id = 'comp4'
        comp_id += '-{}'.format(os.getpid())

        path = os.path.join(self._devkit_path, 'results', 'LSINetOr',
                            '')

        for im_ind, index in enumerate(self.image_index):
            print 'Writing {} VOC results file'.format(index)
            filename = path + index + '.txt'
            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    #elif cls == 'Car_facing' or cls == 'Car_moving_away' or cls == 'Car_looking_left' or cls == 'Car_looking_right':
                    #    write_cls = 'Car'
                    else:
                        write_cls = cls
                    dets = all_boxes[cls_ind][im_ind]

                    if dets == []:
                        continue
                    # the KITTI expects 0-based indices

                    for k in xrange(dets.shape[0]):
                        angle = dets[k, -10:-1]
                        assert len(angle) == 9
                        #print angle
                        angle_bin = np.argmax(angle)
                        #print angle_bin
                        #print dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], dets[k, -1]
                        f.write('{:s} -1 -1 {:.2f} {:.1f} {:.1f} {:.1f} {:.1f} -1 -1 -1 -1000 -1000 -1000 -10 {:.3f}\n'.
                                format(write_cls,
                                       self._orientations[angle_bin],
                                       dets[k, 0], dets[k, 1],
                                       dets[k, 2], dets[k, 3],
                                       dets[k, -10]))

    def competition_mode(self, on):
        print 'Wow, competition mode. Doing nothing...'

if __name__ == '__main__':
    d = datasets.kitti('training')
    res = d.roidb
    from IPython import embed; embed()
