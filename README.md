# How to build

## Nvidia cud
	1.install cuda
	2.install cudnn: https://developer.nvidia.com/rdp/cudnn-download

## Source code compile Tensorflow

So far, I can't build sucessfully based on source code. <br>
Refer: http://www.tensorfly.cn/tfdoc/get_started/os_setup.html	<br>
       https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03 <br>

	if tip: JDK not found, please set $JAVA_HOME.
	sudo apt-get install build-essential openjdk-8-jdk python zip unzip
	
	requirement:	
	$ sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel libcupti-dev
	bazel:(from source code maybe have problem)
	$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	$ sudo apt-get update && sudo apt-get install bazel

## cpp version source code compile Tensorflow

	$ sudo apt-get install openjdk-8-jdk
	$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
	$ curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
	$ sudo apt-get update && sudo apt-get install bazel

	$ sudo apt-get upgrade bazel # this will upgrade Ubuntu OS, could not to do.
	$ cd tensorflow/
	$ ./configure 
	$ bazel build //tensorflow:libtensorflow_cc.so -j 20

**You will see:**

	$ cd bazel-bin/tensorflow
	$ ls
	libtensorflow_cc.so*
	libtensorflow_cc.so-2.params*
	libtensorflow_cc.so.runfiles/
	libtensorflow_cc.so.runfiles_manifest*
	libtensorflow_framework.so*
	libtensorflow_framework.so-2.params*

## cpp version compile Tensorflow test classify

	$ cd tensorflow
	$ wget "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz"
	$ copy inception_v3_2016_08_28_frozen.pb.tar.gz tensorflow/examples/label_image/data/
	$ cd tensorflow/examples/label_image/data/
	$ tar -xf inception_v3_2016_08_28_frozen.pb.tar.gz
	$ cd -
	remove last python part, in the file "tensorflow/examples/label_image/BUILD"
	$ bazel build tensorflow/examples/label_image/...

	$ ./bazel-bin/tensorflow/examples/label_image/label_image

```
2019-01-04 14:43:26.276844: I tensorflow/examples/label_image/main.cc:251] military uniform (653): 0.834305
2019-01-04 14:43:26.276895: I tensorflow/examples/label_image/main.cc:251] mortarboard (668): 0.0218696
2019-01-04 14:43:26.276911: I tensorflow/examples/label_image/main.cc:251] academic gown (401): 0.0103581
2019-01-04 14:43:26.276923: I tensorflow/examples/label_image/main.cc:251] pickelhaube (716): 0.00800825
2019-01-04 14:43:26.276938: I tensorflow/examples/label_image/main.cc:251] bulletproof vest (466): 0.00535093
```

## Apt install Tensorflow

**Scrpt:**

	$ pip install tensorflow-gpu		// GPU=TX1080, CUDA=9.0, cdDNN=7.4
	$ python
	>>> import tensorflow as tf
	>>> tf.enable_eager_execution()
	>>> tf.add(10,2).numpy()
	12

## Train mnist
#### Train

	$ git clone https://github.com/tensorflow/models.git
	$ export PYTHONPATH=$PYTHONPATH:`pwd`/models/
	$ cd models/official/mnist
	$ python mnist.py

	Maybe have errors, if you use *** proxy: ***
	  File "/usr/lib/python2.7/ssl.py", line 830, in do_handshake
	    self._sslobj.do_handshake()
	IOError: [Errno socket error] EOF occurred in violation of protocol (_ssl.c:590)

	Resolve: Because url can't download mnist data, we can manually download, and copy to their direction.
	from shutil import copyfile
	-  #urllib.request.urlretrieve(url, zipped_filepath)
	+  mnist_data_root_path='models/official/mnist/mnist_data/'
	+  copyfile(mnist_data_root_path+filename+'.gz', zipped_filepath)

**Result:**

	$ cd /tmp/mnist_model/
	$ ll
	checkpoint                                  model.ckpt-22800.data-00000-of-00001
	eval                                        model.ckpt-22800.index
	events.out.tfevents.1544772849.xiping-work  model.ckpt-22800.meta
	graph.pbtxt                                 model.ckpt-23400.data-00000-of-00001
	model.ckpt-21600.data-00000-of-00001        model.ckpt-23400.index
	model.ckpt-21600.index                      model.ckpt-23400.meta
	model.ckpt-21600.meta                       model.ckpt-24000.data-00000-of-00001
	model.ckpt-22200.data-00000-of-00001        model.ckpt-24000.index
	model.ckpt-22200.index                      model.ckpt-24000.meta
	model.ckpt-22200.meta

#### Test

	$ export PYTHONPATH=$PYTHONPATH:`pwd`/models/
	$ python mnist_test.py

