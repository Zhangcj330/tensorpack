#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from asyncio import start_unix_server
import itertools
import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
import tqdm
import timeit

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_balloon, register_roof
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow,get_val_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite,gt_mask, apply_masks,draw_outputs, draw_final_outputs_mask)

def unpackbits_masks(masks):
    """
    Args:
        masks (Tensor): uint8 Tensor of shape N, H, W. The last dimension is packed bits.
    Returns:
        masks (Tensor): bool Tensor of shape N, H, 8*W.
    This is a reverse operation of `np.packbits`
    """
    bits = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8)
    unpacked = tf.bitwise.bitwise_and(tf.expand_dims(masks, -1), bits) > 0
    unpacked = tf.reshape(
        unpacked,
        tf.concat([tf.shape(masks)[:-1], [8 * tf.shape(masks)[-1]]], axis=0))
    return unpacked

def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()
    
    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels','gt_masks_packed'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
            'output/masks',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels, gt_masks_packed = dp['image'], dp['gt_boxes'], dp['gt_labels'],dp['gt_masks_packed']
            #  transfer uint8 Tensor of shape N, H, W. to bool Tensor of shape N, H, 8*W.
            unpack_masks =  unpackbits_masks(gt_masks_packed)
            # conver tensor to numpy array
            unpack_masks = unpack_masks.eval(session=tf.compat.v1.Session())
            # modift mask demension to fit the image
            if img.shape[1] > unpack_masks.shape[2]:
                gt_masks = np.zeros((unpack_masks.shape[0],img.shape[0], img.shape[1]))
                gt_masks[:,:,:unpack_masks.shape[2]] = unpack_masks
            elif img.shape[1] < unpack_masks.shape[2]:
                gt_masks = np.zeros((unpack_masks.shape[0],img.shape[0], img.shape[1]))
                gt_masks[:,:,:] = unpack_masks[:,:,:img.shape[1]] 
            else: 
                gt_masks = unpack_masks

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels, masks = pred(img, gt_boxes, gt_labels, gt_masks)
            
            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels, masks = gt_masks)
        
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_output_viz = draw_final_outputs(img, results)

            masked_box_viz = apply_masks(img, final_boxes, masks, final_scores, score_threshold=.5, mask_threshold=0.5)
     
            final_viz = draw_outputs(masked_box_viz, final_boxes, final_scores, final_labels, threshold=0.5)

            viz = tpviz.stack_patches([
                gt_viz, final_output_viz,
                masked_box_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()

def do_visualize_val(model, model_path, nr_visualize=20, output_dir='visualizaion_output_val'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_val_dataflow()
    df.reset_state() 
    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels','gt_masks_packed'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
            'output/masks',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            time_draw = []
            start_it = timeit.default_timer() 
            img, gt_boxes, gt_labels, gt_masks_packed = dp['image'], dp['gt_boxes'], dp['gt_labels'],dp['gt_masks_packed']
            unpack_masks =  unpackbits_masks(gt_masks_packed)
            unpack_masks = unpack_masks.eval(session=tf.compat.v1.Session())
            if img.shape[1] > unpack_masks.shape[2]:
                gt_masks = np.zeros((unpack_masks.shape[0],img.shape[0], img.shape[1]))
                gt_masks[:,:,:unpack_masks.shape[2]] = unpack_masks
            elif img.shape[1] < unpack_masks.shape[2]:
                gt_masks = np.zeros((unpack_masks.shape[0],img.shape[0], img.shape[1]))
                gt_masks[:,:,:] = unpack_masks[:,:,:img.shape[1]] 
            else: 
                gt_masks = unpack_masks
            end_it = timeit.default_timer()
            time_elapse =  end_it - start_it
            time_draw.append(time_elapse)

            time_each = []
            start_it = timeit.default_timer() 
            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels, masks = pred(img, gt_boxes, gt_labels, gt_masks_packed)
            end_it = timeit.default_timer()
            time_elapse =  end_it - start_it
            time_each.append(time_elapse)


            time_anno = []
            start_anno = timeit.default_timer() 
            
            end_anno = timeit.default_timer()
            time_pass =  end_anno - start_anno
            time_anno.append(time_pass)
            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels, masks = gt_masks)
        
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_output_viz = draw_final_outputs(img, results)

            masked_box_viz = apply_masks(img, final_boxes, masks, final_scores, score_threshold=.8, mask_threshold=0.5)
     
            final_viz = draw_outputs(masked_box_viz, final_boxes, final_scores, final_labels, threshold=0.8)

            viz = tpviz.stack_patches([
                gt_viz, final_output_viz,
                masked_box_viz, final_viz], 2, 2)
            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()
            end_anno = timeit.default_timer()
            time_pass =  end_anno - start_anno
            time_anno.append(time_pass)
    time_draw = sum(time_draw)/len(time_draw)
    inference_time = sum(time_each)/len(time_each)
    anno_time = sum(time_anno)/len(time_anno)
    print ('time_draw', time_draw)
    print ('inference_time', inference_time)
    print ('anno_time', anno_time)

def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        
        # get inference results 
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        # check eval_inference_results inside roof dataset
        # give preditions and return COCO metrics
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)

def do_test_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.TEST:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        
        # get inference results 
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        # check eval_inference_results inside roof dataset
        # give preditions and return COCO metrics
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)

def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results) 
    else:
        final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("output.png", viz)
    logger.info("Inference output for {} written to output.png".format(input_file))
    tpviz.interactive_imshow(viz)

def do_test(pred_func, input_file):
    start_image = timeit.default_timer() 
    name = os.path.splitext(os.path.basename(input_file))[0]
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    start_prediction = timeit.default_timer() 
    results = predict_image(img, pred_func)
    end_it = timeit.default_timer()
    time_read.append(end_it - start_image)
    time_predict.append(end_it-start_prediction)
    if cfg.MODE_MASK:
        final = draw_final_outputs_mask(img, results)
    else:
        final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite( "test_output/output_" + name+".png", viz)
    logger.info("Inference output for {} written to test_output/output_{}.png".format(input_file,name))
    

if __name__ == '__main__':
    start = timeit.default_timer()
    print("1 :", start)
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--benchmark', action='store_true', help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--output-pb', help='Save a model to .pb')
    parser.add_argument('--output-serving', help='Save a model to serving file')
    parser.add_argument('--visualize_val', action='store_true', help='visualize intermediate results')
    parser.add_argument('--test', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')
    parser.add_argument('--evaluate_test', help="Run evaluation for test set if test set also have annotations. "
                                           "This argument is the path to the output json evaluation file")                                     

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    register_balloon(cfg.DATA.BASEDIR)
    register_roof(cfg.DATA.BASEDIR)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize or args.visualize_val or args.test:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.visualize:
        do_visualize(MODEL, args.load)

    if args.visualize_val:
        do_visualize_val(MODEL, args.load)
    else:
        predcfg = PredictConfig(
            model=MODEL,
            session_init=SmartInit(args.load),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1])
        
        if args.output_pb:
            ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
        elif args.output_serving:
            ModelExporter(predcfg).export_serving(args.output_serving)
        if args.predict:
            predictor = OfflinePredictor(predcfg)
            for image_file in args.predict:
                do_predict(predictor, image_file)
        elif args.test:
            tester = OfflinePredictor(predcfg)
            if os.path.isdir('test_output'):
                shutil.rmtree('test_output')
            fs.mkdir_p('test_output')
            time_read = []
            time_predict = []
            for image_file in args.test:
                do_test(tester, image_file)
            inference_readtime = sum(time_read)/len(time_read)
            print('Average inference_time including read image: ', inference_readtime)
            inference_predicttime = sum(time_read)/len(time_read)
            print('Average inference_time prediction only: ', inference_predicttime)
            print('maximum inference_time including read image: ', max(time_read))
            print('maximum inference_time prediction only: ', max(time_predict))
            print('minimum inference_time including read image: ', min(time_read))
            print('minimum inference_time prediction only: ', min(time_predict))       
        elif args.evaluate:
            assert args.evaluate.endswith('.json'), args.evaluate
            do_evaluate(predcfg, args.evaluate)
        elif args.evaluate_test:
            assert args.evaluate_test.endswith('.json'), args.evaluate_test
            do_test_evaluate(predcfg, args.evaluate_test)
        elif args.benchmark:
            df = get_eval_dataflow(cfg.DATA.VAL[0])
            df.reset_state()
            predictor = OfflinePredictor(predcfg)
            for _, img in enumerate(tqdm.tqdm(df, total=len(df), smoothing=0.5)):
                # This includes post-processing time, which is done on CPU and not optimized
                # To exclude it, modify `predict_image`.
                predict_image(img[0], predictor)
    stop = timeit.default_timer()
    print('Time: ', stop - start)   
