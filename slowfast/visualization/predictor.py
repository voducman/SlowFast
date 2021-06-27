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
from tracking.sources.tracker import Tracker
from tracking.sources.feature_extractor import FeatureExtractor
from tracking.sources.nn_matching import NearestNeighborDistanceMetric
from tracking.sources.util import create_detections, get_batch_detection

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
        # self.softmax_for_test = torch.nn.Softmax(dim=1)
        self.cfg = cfg_copy
        self.enable_detection = cfg.DETECTION.ENABLE

        if cfg.DETECTION.ENABLE:
            # self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id if cfg.NUM_GPUS > 0 else None)
            self.object_detector = Yolov5Detector(cfg, gpu_id=self.gpu_id if cfg.NUM_GPUS > 0 else None)

        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

        # traking
        metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.3, budget=10)
        self.tracker = Tracker(cfg, metric)
        self.feature_extractor = FeatureExtractor("tracking/checkpoint/mars-small128.pb")

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

            # Tracking
            active_tracks = []
            tracking_frame_interval = int((self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE)/4)
            for i, bboxes in enumerate(task.series_bboxes):
                self.tracker.predict()
                if bboxes is not None:
                    batch_detection_image = get_batch_detection(bboxes, task.frames[i*tracking_frame_interval])
                    human_features = self.feature_extractor(batch_detection_image, batch_size=4)

                    # Add fake detection confidence to bboxes
                    bboxes = torch.cat([bboxes, torch.ones((bboxes.shape[0], 1))], 1)
                    detections = create_detections(bboxes, human_features)
                    self.tracker.update(detections)

                    # Update task.bboxes follow up boxes order of tracker's results
                    if i == 2:
                        active_tracks = [track for track in self.tracker.tracks if track.is_confirmed() and track.time_since_update == 0]
                        updated_bboxes = [t.to_tlbr() for t in active_tracks]
                        task.add_bboxes(torch.tensor(updated_bboxes))
                        # print("Num of tracks:", len(updated_bboxes))

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

        preds = preds.detach()
        for i, pred in enumerate(preds):
            active_tracks[i].update_pred(pred)
            # print("Track active:", active_tracks[i].track_id)
        for track in self.tracker.tracks:
            if track not in active_tracks:
                # print("Track not active:", track.track_id)
                track.update_pred(torch.tensor([]))
        visualize_preds = self.tracker.extract_preds()
        task.add_action_preds(visualize_preds)
        # print("Preds:", visualize_preds.shape)
        # print("Bboxes:", task.bboxes.shape if task.bboxes is not None else None)
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
                task.add_bboxes(pred_boxes.detach().cpu())
            task.add_series_bboxes(pred_boxes.detach().cpu() if (pred_boxes is not None) else None)

        return task


class Yolov5Detector:
    """
    Wrapper around YoloV5 to return the required predicted bounding boxes
    as a ndarray.
    """
    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.input_format = cfg.DEMO.INPUT_FORMAT
        self.conf_thresh = cfg.DEMO.YOLOV5_THRESH

        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.device = torch.device("cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu")
        logger.info("Initialized YoloV5 Object Detection Model.")

        self.predictor = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5{}'.format(cfg.DEMO.YOLOV5_SIZE),
            pretrained=True,
            device=self.device)

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
        batch_frames = [task.frames[idx] for idx in indexes]
        if self.input_format.lower() == 'bgr':
            batch_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
        preds = self.predictor(batch_frames)
        for i, xyxy in enumerate(preds.xyxy):
            conditions = (xyxy[:, 5] == 0) & (xyxy[:, 4] >= self.conf_thresh)
            bboxes = xyxy[conditions][:, :4].int()
            if bboxes.is_cuda:
                bboxes = bboxes.detach().cpu()
            else:
                bboxes = bboxes.detach()

            # keyframe in num_frames * sampling_rate (ex: frames[32])
            if i == 2 and bboxes.shape[0] > 0:
                task.add_bboxes(bboxes)

            task.add_series_bboxes(bboxes if bboxes.shape[0] > 0 else None)
        return task





