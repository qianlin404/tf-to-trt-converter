#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 5/8/19
# Description: Run object detection on a video
# ========================================================

import ctypes
import argparse
import sys
import os
import numpy as np
import tensorrt as trt
import cv2

import utils.inference as inference_utils # TRT/TF inference wrappers
from utils.paths import PATHS # Path management
from datetime import datetime


def parse_commandline_arguments():
    """
    Parse command line arguments
    Returns:

    """
    parser = argparse.ArgumentParser(description="Run object detection on input video frames")
    parser.add_argument("-m", "--model_file", type=str, dest="model_file", metavar="<path>", required=True,
                        help="UFF model file")
    parser.add_argument("-v", "--video_file", type=str, dest="video_file", metavar="<path>", required=True,
                        help="Video file")
    args = parser.parse_args()

    engine_dir = "trt_engines"
    try:
        os.makedirs(engine_dir)
    except FileExistsError:
        pass

    model_name = args.model_file.split('/')[-1].split('.')[0]
    args.engine_path = os.path.join(engine_dir, model_name + ".buf")

    return args


def preprocess_frame(frame):
    """
    preprocess the frame
    Args:
        frame (np.ndarray): frame in np array, HWC format

    Returns:
        frame: normalzied and resized frame
    """
    # resize
    frame = cv2.resize(frame, (300, 300))

    # HWC -> CHW
    frame = frame.transpose([2, 0, 1])

    # normalized
    frame = (2.0 / 255.0) * frame -1

    return frame


def main():
    # Parse arguments
    args = parse_commandline_arguments()


    # Loading FlattenConcat plugin library using CDLL has a side
    # effect of loading FlattenConcat plugin into internal TensorRT
    # PluginRegistry data structure. This will be needed when parsing
    # network into UFF, since some operations will need to use this plugin
    try:
        ctypes.CDLL(PATHS.get_flatten_concat_plugin_path())
    except:
        print(
            "Error: {}\n{}\n{}".format(
                "Could not find {}".format(PATHS.get_flatten_concat_plugin_path()),
                "Make sure you have compiled FlattenConcat custom plugin layer",
                "For more details, check README.md"
            )
        )
        sys.exit(1)

    # build engine
    trt_inference_wrapper = inference_utils.TRTInference(args.engine_path, args.model_file,
                                                         trt.DataType.HALF,
                                                         1)

    inference_time = []
    cnt = 0
    cap = cv2.VideoCapture(args.video_file)

    start = datetime.now()
    success, frame = cap.read()
    frame = preprocess_frame(frame)

    while success:
        cnt += 1
        start_infer = datetime.now()
        trt_inference_wrapper.infer_numpy(frame)
        end_infer = datetime.now()
        inference_time.append((end_infer-start_infer).total_seconds()*1000.0)

        success, frame = cap.read()
    end = datetime.now()

    time_delta = (end-start).total_seconds() * 1000.0
    fps = cnt / time_delta

    print("===============================================================================")
    print("Process time (exclude load model time): {time:.2f}s".format(time=time_delta))
    print("Total #frames: {cnt}".format(cnt=cnt))
    print("Process FPS (exclude load model time): {time:.2f}".format(time=fps))
    print("Average inference time is: {time:.2f}".format(time=np.mean(inference_time)))
    print("===============================================================================")


if __name__ == "__main__":
    main()