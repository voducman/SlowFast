#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import torch
import os
from glob import glob
from tqdm import tqdm

from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


def run_demo(cfg, frame_provider):
    """
    Run demo visualization.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        frame_provider (iterator): Python iterator that return task objects that are filled
            with necessary information such as `frames`, `id` and `num_buffer_frames` for the
            prediction and visualization pipeline.
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES
        if len(cfg.DEMO.LABEL_FILE_PATH) != 0
        else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    async_vis = AsyncVis(video_vis, n_workers=cfg.DEMO.NUM_VIS_INSTANCES)

    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE

    assert (
        cfg.DEMO.BUFFER_SIZE <= seq_len // 2
    ), "Buffer size cannot be greater than half of sequence length."
    num_task = 0
    # Start reading frames.
    frame_provider.start()
    for able_to_read, task in frame_provider:
        if not able_to_read:
            break
        if task is None:
            time.sleep(0.02)
            continue
        num_task += 1

        model.put(task)
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue

    while num_task != 0:
        try:
            task = model.get()
            num_task -= 1
            yield task
        except IndexError:
            continue


def demo(cfg):
    """
    Run inference on an input video or stream from webcam.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    if not os.path.isdir(cfg.DEMO.INPUT_VIDEO):
        raise Exception("Demo input video is currently not a directory.")

    video_paths = glob(os.path.join(cfg.DEMO.INPUT_VIDEO, "*", "*.mp4"))
    conf_output_dir = cfg.OUTPUT_DIR

    for video_path in video_paths:
        video_name = os.path.basename(video_path)
        sub_dir = video_path.split("/")[-2]
        cfg.DEMO.INPUT_VIDEO = video_path
        cfg.OUTPUT_DIR = os.path.join(conf_output_dir, sub_dir)
        cfg.DEMO.OUTPUT_FILE = os.path.join(cfg.OUTPUT_DIR, video_name)

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        # AVA format-specific visualization with precomputed boxes.
        if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
            precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
            precomputed_box_vis()
        else:
            start = time.time()
            print("Start recognizing on video: ", video_name)
            if cfg.DEMO.THREAD_ENABLE:
                frame_provider = ThreadVideoManager(cfg)
            else:
                frame_provider = VideoManager(cfg)

            for task in tqdm(run_demo(cfg, frame_provider)):
                frame_provider.display(task)

            frame_provider.join()
            frame_provider.clean()
            logger.info("Finish demo {} in: {}".format(video_name, time.time() - start))



