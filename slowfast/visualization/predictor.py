#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
import cv2
import numpy as np
import torch
import copy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import slowfast.utils.checkpoint as cu
from slowfast.datasets import cv2_transform
from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.visualization.utils import process_cv2_inputs

logger = logging.get_logger(__name__)


class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        cfg_copy = copy.deepcopy(cfg)
        # to build SlowFast using ResNetBasicHead module
        cfg_copy.DETECTION.ENABLE = False
        self.model = build_model(cfg_copy, gpu_id=gpu_id)
        self.model.eval()
        self.cfg = cfg_copy
        self.enable_detection = cfg.DETECTION.ENABLE

        if cfg.DETECTION.ENABLE:
            self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        if self.enable_detection:
            task = self.object_detector(task)
            person_frames_list = task.extract_person_clip()

        # return if no person detected in keyframe
        if len(person_frames_list) == 0:
            task.add_action_preds(torch.tensor([]))
            return task

        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            person_frames_list_rgb = [[cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames] for frames in person_frames_list]

        person_frames_list_rgb = [
            [cv2_transform.scale(self.cfg.DATA.TEST_CROP_SIZE, f) for f in frames] for frames in person_frames_list_rgb
        ]
        batch_inputs = process_cv2_inputs(person_frames_list_rgb, self.cfg)

        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(batch_inputs, (list,)):
                for i in range(len(batch_inputs)):
                    batch_inputs[i] = batch_inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                batch_inputs = batch_inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )

        preds = self.model(batch_inputs)

        if self.cfg.NUM_GPUS:
            preds = preds.cpu()
            if task.bboxes is not None:
                bboxes = task.bboxes.detach().cpu()

        preds = preds.detach()
        task.add_action_preds(preds)
        task.add_bboxes(bboxes)

        return task


class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(task)
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG)
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = (
            "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"
        )

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        indexes = np.linspace(0, 0.75*len(task.frames), 4, dtype=np.int32)
        for i, idx in enumerate(indexes):
            frame = task.frames[idx]
            outputs = self.predictor(frame)
            # Get only human instances
            mask = outputs["instances"].pred_classes == 0
            pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
            if len(pred_boxes) == 0:
                pred_boxes = None
            if i == 2 and pred_boxes is not None: # middle frame
                task.add_bboxes(pred_boxes)
            task.add_series_bboxes(pred_boxes.clone().detach().cpu() if (pred_boxes is not None) else None)

        return task
