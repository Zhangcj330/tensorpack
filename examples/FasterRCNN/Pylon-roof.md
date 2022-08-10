# Faster R-CNN / Mask R-CNN on Pylon Rooftop

This is a demo on how to train tensorpack's Mask R-CNN on Pylon Rooftop dataset.
For model details, Please see README.md in the folder

## Dependencies

Recommende using "Deep Learning AMI (Ubuntu 16.04) Version 36.0" image to create EC2

- OpenCV, TensorFlow ≥ 1.14 (already install)
- pycocotools/scipy: `for i in cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' scipy; do pip install $i; done`
- pip install tensorpack
- Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/#FasterRCNN) from tensorpack model zoo
- [coco format rooftop data]. Make sure your dataset have the following directory structure:

```
COCOformat/DIR/
  annotations/
    instances_trainXX.json
    instances_valXX.json
  trainXX/
    # image files that are mentioned in the corresponding json
  valXX/
    # image files that are mentioned in corresponding json
```

1. Move the coco format dataset to the right directory:
   In our case, directory path is:

```
DATA.BASEDIR=~/data/pylon
```

2. (included already) Since this dataset is in COCO format, we add a new file [dataset/roof.py](dataset/roof.py) to load the dataset.
   Refer to [dataset/dataset.py](dataset/dataset.py) on the required interface of a new dataset.

3. (included already) Register the names of the new dataset in `train.py` and `predict.py`, by calling `register_roof("/path/to/balloon_roof")`

4. Download a model pretrained on COCO from tensorpack model zoo:

```
wget http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2x.npz
```

5. Start fine-tuning on the new dataset:
   After argument`--config`
   DATA.BASEDIR: the location where our data been stored,
   DATA.VAL: tell model what is our validation data format, which already be set up on strp 2 and 3.
   DATA.TRAIN: tell model what is our training data format. which already be set up on strp 2 and 3.
   TRAIN.EVAL_PERIOD: how often you want to evaluate your model when during the traning process.
   TRAIN.LR_SCHEDULE: when to stop or modify learning rate.
   TRAIN.CHECKPOINT_PERIOD: how often you want to store your model when during the traning process.
   After argument`--load`: the pretrained model when you want to do transfer learning
   argument --logdir : directory to store traning log

```
./train.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
	"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=1 "TRAIN.LR_SCHEDULE=[1000]" \
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
	--load COCO-MaskRCNN-R50FPN2x.npz --logdir train_log/rooftop
```

6. You can train as long as you want, but it only takes **a few minutes** to produce nice results.
   You can visualize the results of the latest model by:

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/checkpoint --predict ~/data/pylon/test-pylon/*.jpg
```

This command will output one image, and highlight the object by making the background black and white. This command will produce images like this in your window:
![ALT](./rooftop-result.png)

7. You can visualize the validation dataset of the latest model by:

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/checkpoint --visualize_val
```

As far, this command will select 100 random images from validation dataset and return 4 visualization result for each iamge.
On the top left graph, it draw the ground true bbox, top right is the all the rooftop bbox prediction with confidence level.
On the bottom left, it onle shows the mask prediction, and the bottom right shows the final bbox prediction with masks
If you want to see the result on training dataset, simply change `--visualize_val` to `--visualize`.
![ALT](./000.png)

8. To evaluation the model, change the --Load to what model you would like to evaluation. it will reture you the COCO object Detection metrics `(mAP(bbox))` `(mAP(segm))`
   the Argument after `--evaluate` is the path you want to store your results. Make sure you create the fold before you run this code.

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/model-8000 --evaluate ~/data/eval/pylon-output.json
```

## - What changes did I make:

1. Add a new file [dataset/roof.py](dataset/roof.py) to load the dataset.
   Edited from [dataset/coco.py](dataset/coco.py). There is some change I made: 1. Change `COCO_id_to_category_id = {} ` + For our own coco-format dataset, change this to an **empty dict**. 2. Change `class_names = ["Roof face"]` 3. Define function register_roof: + change `class_names = ["Roof face"]` + add Pylon Roof datasets a name to the registry, so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.
   ex. `cfg.DATA.TRAIN = roof_train-pylon`, and `cfg.DATA.VAL = roof_val-pylon`

2. Edit [dataset.**init**.py](dataset/__init__.py) file with one line of code ` from .roof import *`.

3. Register the names of the new dataset in `train.py` and `predict.py`, by adding `register_roof("/path/to/balloon_roof")` into those files.

4. the original visualization code didn't support mask. Modify the `do_visualize`, add 4 funtion in [viz.py](viz.py) in order to show mask.

5. Add new argument `--visualize_val` in [predict.py](predict.py) and one funtin ·get_vel_data· in [data.py](data.py) to visualize validation dataset.
