# UNet
Learn UNet to segment image. Resource is from [2]

# Install segmentation_models

	$ git clone https://github.com/qubvel/segmentation_models.git
	$ cd segmentation_models
	$ python3 setup.py Install # set pip mirror to china

	$ python3 train.py

# get_data_coco.py
	
	download coco api: http://cocodataset.org/
	$ cd coco/cocoapi/PythonAPI
	$ sudo python3 setup.py build_ext install # build coco apt and install
	

# Refer: 
[1] https://segmentation-models.readthedocs.io/en/latest/tutorial.html
[2] https://github.com/qubvel/segmentation_models
