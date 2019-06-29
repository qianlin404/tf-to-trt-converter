#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ========================================================
# Author: qianlinliang
# Created date: 5/8/19
# Description: Convert tf ssd_mobilenet_v2_coco model to
#              trt .uff format
# ========================================================

import os
import ctypes
import sys
import argparse

import tensorrt as trt

import utils.model as model_utils # UFF conversion
from utils.paths import PATHS # Path management

# Precision command line argument -> TRT Engine datatype
TRT_PRECISION_TO_DATATYPE = {
    16: trt.DataType.HALF,
    32: trt.DataType.FLOAT
}


def parse_commandline_arguments():
    """Parses command line arguments and adjusts internal data structures."""

    # Define script command line arguments
    parser = argparse.ArgumentParser(description='Convert tf ssd_mobilenet_v2_coco model to uff format')
    parser.add_argument('-f', '--input_file', dest='input_file', type=str, required=True,
                        help="input .pb file to be converted to .uff format")
    parser.add_argument('-p', '--precision', dest='precision', type=int, choices=[32, 16], default=16,
                        help='desired TensorRT float precision to build an engine with')
    parser.add_argument('-b', '--max_batch_size', type=int, default=1, dest='batch_size',
                        help='max TensorRT engine batch size')
    parser.add_argument('-w', '--output_dir', dest='output_dir', default='trt_models',
                        help='sample workspace directory')
    parser.add_argument("--input_shape", nargs='+', type=int, dest="input_shape",
                        help="input shape for this model in CHW format")

    # Parse arguments passed
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    model_name = args.input_file.split('/')[-1].split('.')[0]
    # Fetch TensorRT engine path and data type
    args.trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[args.precision]
    args.output_path = os.path.join(args.output_dir, model_name +
                                    "_bs_{batch_size}_prc_{precision}.uff".format(batch_size=args.batch_size,
                                                                                  precision=args.precision))
    return args


def main():
    # Parse command line arguments
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

    input_shape = (3, 300, 300) if not args.input_shape else args.input_shape
    print("[SSD Mobilenet V2 Converter] Converting {input_file} to UFF format...".format(input_file=args.input_file))
    model_utils.model_to_uff(args.input_file, args.output_path,
                             preprocess_func=model_utils.ssd_mobilenet_v2_unsupported_nodes_to_plugin_nodes,
                             input_shape=input_shape)
    print("[SSD Mobilenet V2 Converter] Convert succeed, output is saved to {output_path}"
          .format(output_path=args.output_path))


if __name__ == '__main__':
    main()
