#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NAME="kitti_frcnn"
GPU_ID="0"
NET="VGG16"
DATASET="kitti"
MODEL_FOLDER="kitti_viewp"

NET_lc=${NET,,}
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  kitti)
    TRAIN_IMDB="kitti_trainsplit"
    TEST_IMDB="kitti_valsplit"
    PT_DIR="kitti"
    ITERS=150000
    ;;
  *)
    echo "No KITTI dataset given: $DATASET"
    exit
    ;;
esac

#../queue_gpu_task.py

# ONLY TRAINING
LOG="experiments/logs/${NAME}_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/${MODEL_FOLDER}/solver.prototxt \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/${NAME}.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

exit
