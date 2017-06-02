This repository contains a modified version of the deep-learning-based object detector *Faster* R-CNN, created by Shaoqing Ren, Kaiming He, Ross Girshick and Jian Sun (Microsoft Research). It is a fork of their python implementation available [here](https://github.com/rbgirshick/py-faster-rcnn).

This version, *lsi-faster-rcnn*, has been developed by Carlos Guindel at the [Intelligent Systems Laboratory](http://uc3m.es/islab) research group, from the [Universidad Carlos III de Madrid](http://www.uc3m.es/home).

Features introduced in this fork include:
* Training (and eventually testing) on the [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php).
* Mixed external/RPN proposals.
* Discrete viewpoint prediction.
* Four-channel input.

The last two features are introduced in two research papers currently accepted for publication. Please check the [citation section](#citing-this-work) for further details.

### Disclaimer

Modifications have been introduced trying to preserve the different functionalities present in the original Faster R-CNN code, which are largely configurable via parameters. Nevertheless, testing has been conducted over a limited set of combinations of parameters; it is not guaranteed in any case the proper operation under all the configuration alternatives. Pull requests fixing unfeasible configuration setups will be welcome.

### License

This work is released under the MIT License (refer to the LICENSE file for details).

### Citing this work

In case you make use of the solutions adopted in this code regarding the viewpoint estimation, please consider citing:

    @inproceedings{Guindel2017ICVES,
        author = {Guindel, Carlos and Mart{\'{i}}n, David and Armingol, Jos{\'{e}} M.},
        booktitle = {IEEE International Conference on Vehicular Electronics
                     and Safety (ICVES)},
        title = {Joint Object Detection and Viewpoint Estimation using CNN features},
        year = {2017},
        note = {Accepted for presentation}
    }

Otherwise, if you use the four-channel input solution, please consider citing:

    @inproceedings{Guindel2017EUROCAST,
        author = {Guindel, Carlos and Mart{\'{i}}n, David and Armingol, Jos{\'{e}} M.},
        booktitle = {EUROCAST 2017, Extended Abstract Book},
        title = {Stereo Vision-Based Convolutional Networks for Object Detection
                 in Driving Environments},
        year = {2017},
        note = {Selected to be included in the Springer LNCS volumes}
    }

You can find the original research paper presenting the Faster R-CNN approach in:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

This fork has been tested with the following GPU devices: NVIDIA Tesla K40, Titan X (Pascal), Titan Xp. We gratefully acknowledge the support of NVIDIA Corporation with the donation of the cited devices to our research group.

For reference, training the VGG16 model uses ~6G of memory in the Titan Xp. Training (and inference) could be performed with less powerful devices using smaller network architectures (ZF, VGG_CNN_M_1024).

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/cguindel/lsi-faster-rcnn.git
  ```
  The `--recursive` flag allows to automatically clone the `caffe-fast-rcnn` submodule. I use [my own fork](https://github.com/cguindel/caffe-fast-rcnn) of [the official repository](https://github.com/rbgirshick/caffe-fast-rcnn). I try to keep it updated with the upstream Caffe repository as far as possible; that is specially relevant when major changes are introduced in some dependency (e.g. cuDNN).

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

  *Ignore notes 1 and 2 if you followed step 1 above.*

  **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need   to manually clone the `caffe-fast-rcnn` submodule:
  ```Shell
  git submodule update --init --recursive
  ```
  **Note 2:** My `caffe-fast-rcnn` submodule is expected to be on the `lsi-faster-rcnn` branch. This will happen automatically *if you followed step 1 instructions*.

3. Edit the line 135 of lib/setup.py to reflect the CUDA compute capability of your GPU. This can be made with an editor (e.g. gedit):
  ```Shell
  cd $FRCN_ROOT/lib
  gedit setup.py
  ```
  The line to be edited is the `arch` flag. For example, for the Titan X Pascal, the following should be writen:
  ```Python
  extra_compile_args={'gcc': ["-Wno-unused-function"],
                              'nvcc': ['-arch=sm_61',
                                       '--ptxas-options=-v',
                                       '-c',
                                       '--compiler-options',
                                       "'-fPIC'"]},
  ```

  Then, build the Cython modules.
  ```Shell
  cd $FRCN_ROOT/lib
  make
  ```

4. Build Caffe and pycaffe
  ```Shell
  cd $FRCN_ROOT/caffe-fast-rcnn
  # Now follow the Caffe installation instructions here:
  #   http://caffe.berkeleyvision.org/installation.html

  # If you're experienced with Caffe and have all of the requirements installed
  # and your Makefile.config in place, then simply do:
  make -j8 && make pycaffe
  ```

5. If you want to run our demo, please download the trained models:
  ```Shell
  cd $FRCN_ROOT
  ./data/scripts/fetch_lsi_models.sh
  ```

  This will populate the `$FRCN_ROOT/data` folder with `lsi_models`. These models were trained on KITTI.

6. Our demo also requires to found the KITTI object dataset in `$FRCN_ROOT/data/kitti/images`. You will need to download the dataset from [their site](http://www.cvlibs.net/datasets/kitti/) and then create a symbolic link to `$FRCN_ROOT/data/kitti/images`:

  ```Shell
  ln -s $PATH_TO_OBJECT_KITTI_DATASET $FRCN_ROOT/data/kitti/images
  ```

  Please note that `PATH_TO_OBJECT_KITTI_DATASET` must contain, at least, the `testing` folder with the left color images (`image_2`) in it.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo_viewp.py
```
The demo performs Faster R-CNN detection and viewpoint inference using a VGG16 network trained for detection on the [KITTI Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php).

<!-- ### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models -->

<!-- ### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA. -->

<!-- ### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
``` -->
