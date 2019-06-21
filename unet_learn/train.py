import sys
# sys.path.insert(0, '../segmentation_models')
sys.path.insert(0, '../segmentation_models')

import tensorflow as tf
import numpy as np
import cv2

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

import os
def get_data(imgDir, maxNum=10, valMaxNum=10):
	img_path = imgDir + "/images"
	mask_path = imgDir + "/mask"
	img_files = [ f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path,f)) ]
	# mask_files = [ f for f in os.listdir(mask_path) if os.path.isfile(os.path.join(mask_path,f)) ]

	# print(len(img_files))
	# print(img_files[0])
	# exit(0)

	train_data = []
	train_mask_data = []

	val_data = []
	val_mask_data = []

	# ttt=np.array()
	print("Start load images ...")
	total_num = len(img_files)
	for idx, fn in enumerate(img_files):
		print("load progress = [", total_num, ",", idx, "]", end=" ")
		img = cv2.imread(img_path + "/" + fn)
		if img is None:
			print("Can't imread: ", img_path + "/" + fn)
			continue
		mask = cv2.imread(mask_path + "/" + fn, 0)
		if mask is None:
			print("Can't imread: ", mask_path + "/" + fn)
			continue
		mask=mask.reshape(mask.shape[0],mask.shape[1],1)

		img=cv2.resize(img, (224,224))
		mask=cv2.resize(mask, (224,224)).reshape(224,224,1)
		mask=mask>128

		img = img.astype('float32')/255.0
		mask = mask.astype('float32')
		# print(mask.shape)
		# cv2.imwrite("xx.png", mask)
		# exit(0)

		if idx < maxNum:
			print("train")
			train_data.append(img)
			train_mask_data.append(mask)
		elif idx < maxNum + valMaxNum:
			print("validate")
			val_data.append(img)
			val_mask_data.append(mask)
		else:
			break

	print("Load images finish")
	all_train=(np.array([i for i in train_data]), np.array([i for i in train_mask_data]))
	all_val=(np.array([i for i in val_data]), np.array([i for i in val_mask_data]))
	return all_train, all_val

def train_unet_mobilenetv2(saveModelFn, tensorboardPath):
	# train_imgDir = "/home/xiping/mydisk2/imglib/my_imglib/coco/train2014_person"
	train_imgDir = "/coco/train2014_person"
	(train_data, train_mask_data), (val_data, val_mask_data)=get_data(train_imgDir, maxNum=12000, valMaxNum=1000)

	# print(train_data.shape)
	# print(mask_data.shape)
	# print(mask_data[0])
	# cv2.imwrite("xx.bmp", mask_data[1]*255)
	# exit(0)

	print("================================")
	BACKBONE = 'mobilenetv2'
	# define model
	model = Unet(BACKBONE, classes=1, 
		input_shape=(224, 224, 3),	# specific inputsize for callback save model
		activation='sigmoid', #sigmoid,softmax
		encoder_weights='imagenet')

	# Show network structure.
	# model.summary()

	model.compile('Adam', loss='jaccard_loss', metrics=['iou_score'])
	# model.compile('SGD', loss="bce_dice_loss", metrics=["dice_score"])
	# model.compile('SGD', loss="bce_jaccard_loss", metrics=["iou_score"])
	# model.compile('adam', loss="binary_crossentropy", metrics=["iou_score"])
	
	checkpointer = ModelCheckpoint(
		filepath="weights.epoch={epoch:02d}-val_loss={val_loss:.2f}-val_iou_score={val_iou_score:.2f}.hdf5", 
		verbose=1, save_best_only=True)

	print("================================")
	print("Start train...")
	# fit model
	# if you use data generator use model.fit_generator(...) instead of model.fit(...)
	# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
	model.fit(
		x=train_data,
		y=train_mask_data,
		batch_size=32,
		epochs=200,
		validation_data=(val_data, val_mask_data),# callback save middle model need input val data
		callbacks=[TensorBoard(log_dir=tensorboardPath), checkpointer]
	)

	model.save(saveModelFn)

def overlap_mask(src, mask, alpha=0.6):
	m=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
	return cv2.addWeighted(src, alpha, m, 1-alpha, 0)

import keras
def test_image(saveModelFn):
	print("-------------start load_model")
	model=keras.models.load_model(saveModelFn)

	print("-------------start compile")
	model.compile('Adam', loss='jaccard_loss', metrics=['iou_score'])

	print("-------------start read data")
	train_imgDir = "/coco/train2014_person"
	(train_data, mask_data), (val_data, val_mask_data)=get_data(train_imgDir, maxNum=10)

	print("-------------start predict")
	pmask=model.predict(train_data[:1])

	print("-------------start save result")
	src=(train_data[:1]*255).reshape(224,224,3)
	cv2.imwrite("test.png", src)

	rmask=(mask_data[:1]*255).reshape(224,224,1)
	cv2.imwrite("rmask.png", overlap_mask(src, rmask, 0.7))

	pmask=(pmask*255).reshape(224,224,1)
	cv2.imwrite("pmask.png", overlap_mask(src, pmask, 0.7))
	print("test_image finish")

if __name__ == '__main__':
	tensorboardPath="./unet_mobilenetv2_tensorboard"
	saveModelFn="person_segment_unet_moblienetv2.h5"
	train_unet_mobilenetv2(saveModelFn, tensorboardPath)
	saveModelFn="./bk.h5"
	# test_image(saveModelFn)