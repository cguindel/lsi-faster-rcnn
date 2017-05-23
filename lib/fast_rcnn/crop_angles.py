import caffe
import numpy as np
from fast_rcnn.config import cfg

class CropAnglesLayer(caffe.Layer):

    #bottom[0] = 72x vector - output of cnn
    #bottom[1] = 72x vector - weights

    def setup(self, bottom, top):

        self._viewp_bins = cfg.VIEWP_BINS

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs")

    def reshape(self, bottom, top):

        top[0].reshape(1, self._viewp_bins)

    def forward(self, bottom, top):
        np.set_printoptions(threshold=np.nan)
        original = bottom[0].data
        weights = bottom[1].data

        cropped = np.zeros((weights.shape[0], self._viewp_bins), \
                            dtype=np.float32)

        for nrow, row in enumerate(original):
            cropped[nrow, :] = row[weights[nrow]>0]

        top[0].reshape(*cropped.shape)
        top[0].data[...] = cropped

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
          raise Exception("Weights cannot be propagated down")

        original = bottom[0].data
        weights = bottom[1].data
        topdiff = top[0].diff[...]

        prop = np.zeros_like(original)
        for nrow, row in enumerate(prop):
          row[weights[nrow]==1] = topdiff[nrow,:]

        bottom[0].reshape(*prop.shape)
        bottom[0].diff[...] = prop
    
