import cv2
import os
import numpy as np
from .detection import Detection

IMAGE_SHAPE = (128, 64)

def bgr_resize_to_rgb(input_image):
    """Resize image to same with input dimenssion of WRN network (128,64,3)
    Parameters
    ----------
    input_image: 3d-array

    Returns
    -------
    output_resize: 3d-array
    """
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    output_resize = cv2.resize(input_image, IMAGE_SHAPE, interpolation=cv2.INTER_AREA)
    return output_resize


def resize(input_image):
    return cv2.resize(input_image, IMAGE_SHAPE, interpolation=cv2.INTER_AREA)


def gather_sequence_info(video_dir, video_capture, image_encoder):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    video_dir : str
        Path to the video directory.
    video_capture : instance of cv2.VideoCapture
        a instance of cv2.VideoCapture class
    image_encoder: instance of ImageEncoder

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * video_name: Name of the sequence
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.
    """

    update_ms = 1000 / int(video_capture.get(cv2.CAP_PROP_FPS))

    feature_dim = image_encoder.feature_dim
    image_size = image_encoder.image_shape

    seq_info = {
        "video_name": os.path.basename(video_dir),
        "image_size": image_size,
        "min_frame_idx": 0,
        "max_frame_idx": int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(xyxy_list, features):
    if len(xyxy_list) == 0 or xyxy_list is None:
        return []

    detections = []
    for i, coor in enumerate(xyxy_list):
        x = coor[0]
        y = coor[1]
        w = coor[2] - coor[0]
        h = coor[3] - coor[1]
        conf = coor[4]
        detections.append(Detection((x,y,w,h), conf, features[i]))

    return detections


def get_batch_detection(xyxy_list, frame):
    detection_batch = []
    for coor in xyxy_list:
        coor = coor.numpy()
        # frame is (h,w,c)
        sub_image = frame[coor[1]:coor[3], coor[0]:coor[2], :].copy()
        resize_image = resize(sub_image)
        # convert (h,w,c) -> (w,h,c)
        resize_image = np.transpose(resize_image, (1,0,2))
        detection_batch.append(resize_image)
    return np.asarray(detection_batch)


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        >>> boxes = [d.roi for d in detections]
        >>> scores = [d.confidence for d in detections]
        >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick
