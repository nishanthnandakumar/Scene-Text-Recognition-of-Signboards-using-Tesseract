# USAGE
# python intersection_over_union.py

# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

# define the list of example detections
examples = [
	Detection("ktx.jpg", [59,52,261,298], [49,63,264,143]),
	Detection("029.jpg", [27,54,454,163], [1,56,472,159]),
	Detection("056.JPG", [81,159,545,242], [53,162,588,243]),
	Detection("065.jpg", [295,132,482,190], [283,140,492,191]),
	Detection("104.jpg", [70,86,553,196], [83,109,572,202]),

	Detection("137.JPG", [222,262,414,300], [223,264,414,297]),
	Detection("139.JPG", [82,245,541,322], [47,251,588,314]),
	Detection("080116-0064.jpg", [311,225,541,337], [294,228,562,334]),

	Detection("DSC02408.JPG", [58,260,569,350], [0,233,640,353]),
	Detection("DSC02984.JPG", [21,96,631,206], [0,115,640,187]),
	Detection("DSC03167.JPG", [207,294,396,355], [218,302,418,342]),
	Detection("DSC03279.JPG", [140,208,329,260], [137,215,346,255]),

	Detection("DSC03300.JPG", [164,118,482,247], [142,121,512,251]),
	Detection("ktx.jpg", [59,52,261,155], [49,63,264,143]),
	Detection("DSC03896.JPG", [145,227,534,394], [116,252,588,375]),

	Detection("DSC02674.JPG", [229,197,620,307], [198,189,640,287]),
	Detection("P090911067.jpg", [72,197,433,288], [183,204,454,302]),


	Detection("P090903050.jpg", [34,71,517,190], [0,74,576,182])]

# loop over the example detections
for detection in examples:
	# load the image
	image = cv2.imread(detection.image_path)

	# draw the ground-truth bounding box along with the predicted
	# bounding box
	cv2.rectangle(image, tuple(detection.gt[:2]),
		tuple(detection.gt[2:]), (0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.pred[:2]),
		tuple(detection.pred[2:]), (0, 0, 255), 2)

	# compute the intersection over union and display it
	iou = bb_intersection_over_union(detection.gt, detection.pred)
	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou))

	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
