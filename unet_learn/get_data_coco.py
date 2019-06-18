

import cv2 
import pycocotools.mask as mask
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np

# brief Parsing coco dataset
#	http://cocodataset.org/#download
# 	refer: https://github.com/nightrome/cocostuffapi/blob/master/PythonAPI/pycocoDemo.ipynb
# dataDir: root path
# dataType: "train2014", "val2014", "test2014"
# return data, mask
def get_data_coco(dataDir, dataType="train2014"):
	annType = ['segm', 'bbox', 'keypoints']
	annType = annType[0]  # specify type here
	prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'

	annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
	coco=COCO(annFile)
	# anns = coco.anns
	
	# display COCO categories and supercategories
	cats = coco.loadCats(coco.getCatIds())
	nms=[cat['name'] for cat in cats]
	print('COCO categories: \n{}\n'.format(' '.join(nms)))

	nms = set([cat['supercategory'] for cat in cats])
	print('COCO supercategories: \n{}'.format(' '.join(nms)))

	# get all images containing given categories, select one at random
	# catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
	catIds = coco.getCatIds(catNms=['person'])
	imgIds = coco.getImgIds(catIds=catIds)
	# imgIds = coco.getImgIds(imgIds=[324158])

	img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

	print("Image total = ", len(imgIds))
	for idx in range(len(imgIds)):
		img = coco.loadImgs(imgIds[idx])[0]

		img_fn = dataDir + "/" + dataType + "/" + img["file_name"]
		show_img = cv2.imread(img_fn)

		# print(img_fn)
		# cv2.imshow("x", show_img)
		# cv2.waitKey(0)

		annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
		anns = coco.loadAnns(annIds)
		mask_img = None
		for i in range(len(anns)):
			# print(anns[i]["segmentation"])
			# print(anns[i]["image_id"])
			mask = coco.annToMask(anns[i])
			if mask_img is None:
				mask_img = mask
			else:
				mask_img = mask_img | mask

		bgrmask = cv2.cvtColor(mask_img*255, cv2.COLOR_GRAY2BGR)
		cv2.imshow("mask", bgrmask | show_img)
		cv2.waitKey(0)
	return None, None

def main():
	dataDir="/home/xiping/mydisk2/imglib/my_imglib/coco"
	dataType = "val2014"
	val_data, val_mask = get_data_coco(dataDir, dataType)

if __name__ == '__main__':
	main()