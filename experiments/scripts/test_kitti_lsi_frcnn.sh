#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

NAME="kitti_frcnn"
GPU_ID="0"
NET="VGG16"
NET_lc=${NET,,}
DATASET="kitti"
MODEL_FOLDER="kitti_viewp"
NET_FINAL="data/lsi_models/vgg16.caffemodel"

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  kitti)
    TRAIN_IMDB="kitti_trainsplit"
    TEST_IMDB="kitti_valsplit"
    PT_DIR="kitti"
    ;;
  *)
    echo "No dataset given: $DATASET"
    exit
    ;;
esac

LOG="experiments/logs/test_${NAME}_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/${MODEL_FOLDER}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${NAME}.yml \
  --num_dets -1 \
  --thresh 0 \
  #--vis
  ${EXTRA_ARGS}
