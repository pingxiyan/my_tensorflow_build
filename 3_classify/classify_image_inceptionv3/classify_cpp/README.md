# REFER

	https://tuanphuc.github.io/standalone-tensorflow-cpp/
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/cmake/tf_label_image_example.cmake
    
    1. build tensorflow, refer:tensorflow_build/README.md
    2. copy headers and libraries to current project.

# Dependencies

	$ sudo apt install libeigen3-dev
# Copy headers and libraries

	$ mkdir -p include/third_party
	$ mkdir lib
	
	$ cp ~/opensource/tensorflow/tensorflow/examples/label_image/main.cc ./
	$ cp 
