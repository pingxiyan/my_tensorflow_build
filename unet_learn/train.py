import sys
# sys.path.insert(0, '../segmentation_models')
sys.path.insert(0, '../segmentation_models')

import tensorflow as tf
import numpy as np
import cv2 as cv

from segmentation_models import Unet

def main():
	print("==============")
	model = Unet('mobilenetv2', classes=2, activation='softmax')

if __name__ == '__main__':
	main()