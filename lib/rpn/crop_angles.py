import caffe
import numpy as np
from fast_rcnn.config import cfg

class CropAnglesLayer(caffe.Layer):

    #bottom[0] = 72x vector - output of cnn
    #bottom[1] = 72x vector - weights

    def setup(self, bottom, top):
        # check input pair
        self._viewp_bins = cfg.VIEWP_BINS
        assert (self._viewp_bins == 8)
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        #if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #self.original = np.zeros_like(bottom[0].data, dtype=np.float32)
        #self.weights = np.zeros_like(bottom[1].data, dtype=np.float32)
        #self.cropped = np.zeros((bottom[1].data.shape[0], 8), dtype=np.float32)

        # loss output is scalar
        top[0].reshape(1, self._viewp_bins)

    def forward(self, bottom, top):
        np.set_printoptions(threshold=np.nan)
        original = bottom[0].data
        weights = bottom[1].data
        #gt = bottom[2].data

        #print original[10,:]
        #print gt[10,:]

        #self.weights = np.empty_like(weights)
        #self.original = np.empty_like(original)

        #np.copy(self.weights, weights)
        #np.copy(self.original, original)

        cropped = np.zeros((weights.shape[0], self._viewp_bins), dtype=np.float32)



        #print original[0,:]
        #self.diff[...] = bottom[0].data - bottom[1].data
        #top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
        for nrow, row in enumerate(original):
          #nze = np.argmax(weights)/8
          #for i in xrange(8):
          #  if i == gt[nrow]:
              cropped[nrow, :] = row[weights[nrow]>0]

        #print cropped[0,:]

        #print cropped[10,:]
        top[0].reshape(*cropped.shape)
        top[0].data[...] = cropped


    def backward(self, top, propagate_down, bottom):
        print 'hola'
        if propagate_down[1]:
          raise Exception("Como vas a propagar los pesos, hijoputa")

        original = bottom[0].data
        weights = bottom[1].data
        topdiff = top[0].diff[...]

        #print original[0,:]
        #print self.original[0,:]

        prop = np.zeros_like(original)
        for nrow, row in enumerate(prop):
          row[weights[nrow]==1] = topdiff[nrow,:]

        #print 'BACKWARDS'
        #print top[0].diff[...]
        bottom[0].reshape(*prop.shape)
        bottom[0].diff[...] = prop
        #print prop[0,:]

        #for i in range(2):
        #    if not propagate_down[i]:
        #        continue
        #    if i == 0:
        #        sign = 1
        #    else:
        #        sign = -1
        #    bottom[i].diff[...] = sign * self.diff / bottom[i].num
