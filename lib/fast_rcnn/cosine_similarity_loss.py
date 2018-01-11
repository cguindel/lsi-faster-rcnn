# --------------------------------------------------------
# LSI-Faster R-CNN
# Original work Copyright (c) 2015 Microsoft
# Modified work Copyright 2018 Carlos Guindel
# Licensed under The MIT License [see LICENSE for details]
# Written by Carlos Guindel
# --------------------------------------------------------

import caffe
import numpy as np
import math
import yaml

class CosineSimilarityLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        layer_params = yaml.load(self.param_str_)
        self._ignore_label = layer_params['ignore_label']

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.cosdiff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        gt_angles = bottom[0].data
        det_angles = bottom[1].data
        self.diff[...] = gt_angles - det_angles
        self.diff[gt_angles==self._ignore_label] = 0
        self.cosdiff[...] = (1+np.cos(self.diff)) / 2.
        top[0].data[...] = np.sum(1-self.cosdiff) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * np.sin(self.diff) / bottom[i].num / 2.
