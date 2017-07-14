import cython
from cython.parallel import prange, parallel

@cython.boundscheck(False)
def bbox_overlaps(double [:, :] boxes, double [:, :] query_boxes, double [:,:] overlaps, int method):
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef double iw, ih, box_area, boxes_box_area
    cdef double ua
    cdef unsigned int k, n
    cdef int met

    with nogil, parallel():
        for k in prange(K, schedule='dynamic'):
            box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            )
            for n in prange(N, schedule='dynamic'):
                iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                    )
                    if ih > 0:
                        boxes_box_area = (
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1)
                        )
                        if met==0:
                            ua = float(
                                boxes_box_area +
                                box_area - iw * ih
                            )
                            overlaps[n, k] = iw * ih / ua
                        else:
                            ua = float(
                                min(boxes_box_area, box_area)
                            )
                            overlaps[n, k] = iw * ih / ua
