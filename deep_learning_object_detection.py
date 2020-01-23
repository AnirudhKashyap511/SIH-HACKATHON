# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import sys

args = {'prototxt': 'MobileNetSSD_deploy.prototxt.txt', 'model': 'MobileNetSSD_deploy.caffemodel',
		'confidence': 0.2, 'video': sys.argv[1]}

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv2.VideoCapture(args['video'])

c = ["bicycle", "bird", "bus", "car", "cat", "cow", 
	 "dog", "horse", "motorbike", "person"]

f = 0

while (1):
	_, img = cap.read()
	v = 0
	p = 0	
	img = cv2.resize(img, (640, 480))
	(h, w) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()	
	f += 1
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			if CLASSES[idx] in c:
				if CLASSES[idx] == "person":
					p += 1
				elif CLASSES[idx] == "bus" or CLASSES[idx] == "car" or CLASSES[idx] == "motorbike":
					v += 1  	
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				#print("[INFO] {}".format(label))
				cv2.rectangle(img, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(img, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	if f % 300 == 0:
		f, v, p = 0, 0, 0  			
	print("person: {}, veichle: {}".format(p, v))			
	cv2.imshow('img', img)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()