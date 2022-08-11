# Faster R-CNN / Mask R-CNN on Pylon Rooftop

This is a demo on how to train Tensorpack's Mask R-CNN on the Pylon Rooftop dataset.
For model details, Please see README.md in the folder

### Recommend using the "Deep Learning AMI (Ubuntu 16.04) Version 36.0" image to create EC2.

Activate TensorFlow 1.15 virtual enviroment by `source activate tensorflow_p36`
Get Tensorpack repository from `git clone -b pylon https://github.com/Zhangcj330/tensorpack.git`

## Dependencies

- OpenCV, TensorFlow ≥ 1.14 (already install)
- pycocotools/scipy: `for i in cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' scipy; do pip install $i; done`
- `pip install tensorpack`
- Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/#FasterRCNN) from tensorpack model zoo
  We used [COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz](http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz) as our pre-trained model to start our transfer leaning job.
- [COCO format rooftop data](). Make sure your dataset has the following directory structure:

```
COCOformat/DIR/
  annotations/
    instances_trainXX.json
    instances_valXX.json
  trainXX/
    # image files that are mentioned in the corresponding JSON
  valXX/
    # image files that are mentioned in the corresponding JSON
```

## Traning Process

the model we currently using is ResNet-101. Compare to ResNet-50, R101 is more complicated, the inference time will be longer, but more ideal prediction. The average inference time in a g4dn.xlarge instance will be around 0.35 sec.

For more detail about ResNet-101 and its configration, see [document]()

### Step for training

1. Move the coco format dataset to the right directory:
   In our case, the default directory path is: `DATA.BASEDIR=~/data/pylon`, so make sure to put your roof dataset into this path or you could specify the path in the configaration argument.

2. (included already) Since this dataset is in COCO format, we add a new file [dataset/roof.py](dataset/roof.py) to load the dataset.
   Refer to [dataset/dataset.py](dataset/dataset.py) on the required interface of a new dataset.

3. (included already) Register the names of the new dataset in `train.py` and `predict.py`, by calling `register_roof("/path/to/balloon_roof")`

4. Download a model pre-trained on COCO from the Tensorpack model zoo:

```
wget http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz
```

5. Start fine-tuning the model with the roof dataset:
   After argument`--config`
   DATA.BASEDIR: the location where our data has been stored,
   DATA.VAL: tell the model what is our validation data format, which already been set up in steps 2 and 3.
   DATA.TRAIN: tell the model what is our training data format. which already be set up in steps 2 and 3.
   TRAIN.EVAL_PERIOD: how often you want to evaluate your model when during the training process.
   TRAIN.LR_SCHEDULE: when to stop or modify learning rate.
   TRAIN.CHECKPOINT_PERIOD: how often you want to store your model when during the training process.
   After argument`--load`: the pre-trained model when you want to do transfer learning.
   argument `--logdir` : directory to store training log and trained models.

```
./train.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
	"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=1 "TRAIN.LR_SCHEDULE=[5000]" \
	FPN.CASCADE=True 'BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]'\
	FPN.NORM=GN BACKBONE.NORM=GN  BACKBONE.FREEZE_AT=0\
	FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head\
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,800]" TRAIN.CHECKPOINT_PERIOD=5 DATA.NUM_WORKERS=1 \
	--load COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz  --logdir train_log/rooftop
```

## Model Inference

1. You can visualize the results of the latest model by:

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/checkpoint --test ~/data/pylon/test-pylon/*.jpg
```

This command will output predicted images in the directory `test-output/`, give each mask a random colour and highlight the object by making the background black and white. This command will produce images like this in your window:
![ALT](./output_1.png)

2. You can visualize the validation dataset of the latest model by:

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/checkpoint --visualize_val
```

The command will select 100 images from the validation dataset and return 4 visualization results for each image.
On the top left graph, it draws the ground true b-box, top right is the all the rooftop box prediction with the confidence level.
On the bottom left, it only shows the mask prediction, and the bottom right shows the final b-box prediction with masks
If you want to see the result on the training dataset, simply change `--visualize_val` to `--visualize`.
![ALT](./000.png)

3. To evaluate the model, change the --Load to what model you would like to evaluate. it will return you the COCO object Detection metrics `(mAP(bbox))` `(mAP(segm))` on validation set. The Argument after `--evaluate` is the path you want to store your results.

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/model-8000 --evaluate Pylon_val_eval_output.json
```

4. (optional) Because our test dataset also have labelled annotation. we can also evaluate the COCO object Detection metrics on the test dataset.

```
./predict.py --config DATA.BASEDIR=~/data/pylon MODE_FPN=True \
"DATA.VAL=('roof_val-pylon',)"  "DATA.TRAIN=('roof_train-pylon',)" \
--load train_log/rooftop/model-35500 --evaluate_test Pylon_test_eval_output.json
```

## Things you might want to do to improve

1. Fine-tunning `RPN.PROPOSAL_NMS_THRESH` before training.
   NMS is short for non maximum suppressio, a way to select one entity out of many overlapped perdiction.
2. Fine-tunning `TEST.FRCNN_NMS_THRESH` and `TEST.RESULT_SCORE_THRESH_VIS` when inferencing, change those 2 numbers will aslo give you different COCO matric. but I don't think you could only trust the metrics.
   TEST.FRCNN_NMS_THRESH: it will ignore the bboxs who are interacting IOU > FRCNN_NMS_THRESH.
   Smaller the threshold will prevent b-box and mask overlapping but may ignore some small pieces of roof that just happens to overlap with the other boxes.
   TEST.RESULT_SCORE_THRESH_VIS: it is the minimum confidence level for b-box detection.
3. Mannully modify the threshold in funtion `_paste_mask` in [eval.py](eval.py) when inferencing.
   In line 87, ` return (res > 0.5).astype('uint8')`
   The higher the therhold, the more strict for a pixel to be predited as a mask. resulting the edge cutting of the masks.
4. Mannully modify the threshold in funtion `mask_regularization` in [viz.py](viz.py) when inferencing.
   In line 25, play with the number to get better performace.
   the smaller the number, the More irregular shape.
5. Write a logic to avoid overlapping mask before draw contours

If you are interested in the training log and metrics for each training epoch, have a look at the [my_training_logs](my_training_logs)

## COCO meteics results

## - What changes did I make:

1. Add a new file [dataset/roof.py](dataset/roof.py) to load the dataset.
   Edited from [dataset/coco.py](dataset/coco.py). There is some change I made: 1. Change `COCO_id_to_category_id = {} ` + For our own coco-format dataset, change this to an **empty dict**. 2. Change `class_names = ["Roof face"]` 3. Define function register_roof: + change `class_names = ["Roof face"]` + add Pylon Roof datasets a name to the registry, so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.
   ex. `cfg.DATA.TRAIN = roof_train-pylon`, and `cfg.DATA.VAL = roof_val-pylon`

2. Edit [dataset.**init**.py](dataset/__init__.py) file with one line of code ` from .roof import *`.

3. Register the names of the new dataset in `train.py` and `predict.py`, by adding `register_roof("/path/to/balloon_roof")` into those files.

4. the original visualization code didn't support masks. Modify the `do_visualize`, add 4 functions in [viz.py](viz.py) in order to show mask.

5. Add new argument `--visualize_val` in [predict.py](predict.py) and one function ·get_vel_data· in [data.py](data.py) to visualize validation dataset.

Updated Aug 11:

1. Switch the model from ResNet-50 to ResNet-101, and change the configuration as required.
2. Overlapping mask.
3. Add function mask_regularization in [viz.py](viz.py) in order to draw straight lines for mask contours.
   for [more detail](https://stackoverflow.com/questions/58736927/how-to-find-accurate-corner-positions-of-a-distorted-rectangle-from-blurry-image)
4. Add new argument `--evaluate_test` in [predict.py](predict.py) to evaluate the test set.
