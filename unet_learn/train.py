import sys
# sys.path.insert(0, '../segmentation_models')
sys.path.insert(0, '../segmentation_models')

import tensorflow as tf
import numpy as np
import cv2 as cv

from segmentation_models import Unet

# # import segmentation_models.unet as unet
# # from segmentation_models.unet import Unet

# def main():
# 	print("==============")
# 	# model = unet()

# if __name__ == '__main__':
# 	main()