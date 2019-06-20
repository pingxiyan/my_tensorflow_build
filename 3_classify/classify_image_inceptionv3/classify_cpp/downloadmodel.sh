#!/bin/bash

echo "start model"
echo "refer:tensorflow/tensorflow/examples/label_image"

wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
tar -xf inception_v3_2016_08_28_frozen.pb.tar.gz
