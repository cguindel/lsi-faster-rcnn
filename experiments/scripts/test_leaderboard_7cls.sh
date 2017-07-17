#!/bin/bash
set -x
set -e

# Parameters
NAME="leaderboard_7cls"
GPU_ID="1"
PT_DIR="kitti"
NET="VGG16"
TRAIN_IMDB="kitti_training"
TEST_IMDB="kitti_test_testing"
NET_FINAL="output/${NAME}/${TRAIN_IMDB}/vgg16_faster_rcnn_iter_150000.caffemodel"

# Script
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/test_${NAME}_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export PYTHONUNBUFFERED="True"

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${PT_DIR}/${NET}/${NAME}/test.prototxt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg experiments/cfgs/${NAME}.yml \
  --num_dets -1 \
  --thresh 0 \
  ${EXTRA_ARGS}
