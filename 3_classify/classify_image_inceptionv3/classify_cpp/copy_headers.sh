#!/bin/sh

TENSORFLOW_DIR=/home/xiping/opensource/tensorflow
echo "delete old headers"
rm -rf include/third_party
mkdir -p include/third_party

echo "start copy new headers"
cp -r $TENSORFLOW_DIR/tensorflow include/third_party/
cp -r $TENSORFLOW_DIR/bazel-genfiles/tensorflow include/third_party/
cp -r $TENSORFLOW_DIR/third_party/eigen3 include/third_party/

cp -r $TENSORFLOW_DIR/tensorflow/contrib/makefile/downloads/absl/absl include/third_party/
cp -r $TENSORFLOW_DIR/tensorflow/contrib/makefile/gen/protobuf/include/google include/third_party/
