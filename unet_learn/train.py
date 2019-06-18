import sys
# sys.path.insert(0, '../segmentation_models')
sys.path.insert(0, '../segmentation_models')

import tensorflow as tf
import numpy as np
import cv2 as cv

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

def get_data(path, train=True):
	return None, None

def main():
	print("================================")
	BACKBONE = 'mobilenetv2'
	# define model
	model = Unet(BACKBONE, classes=2, 
		input_shape=(None, None, 3),
		activation='softmax', 
		encoder_weights='imagenet')

	model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

	print("================================")
	print("Start train...")
	# fit model
	# if you use data generator use model.fit_generator(...) instead of model.fit(...)
	# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
	# model.fit(
	#     x=x_train,
	#     y=y_train,
	#     batch_size=16,
	#     epochs=100,
	#     validation_data=(x_val, y_val),
	# )
if __name__ == '__main__':
	main()