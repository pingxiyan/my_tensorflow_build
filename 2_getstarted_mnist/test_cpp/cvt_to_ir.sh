#!/bin/bash

mo="/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py"
model="../model_pb/lenet.pd"

${mo} --input_model ${model} --data_type FP32 --disable_nhwc_to_nchw --input_shape "(1,28,28,1)"


