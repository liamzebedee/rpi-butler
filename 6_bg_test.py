import picamera
from time import sleep
import os
import sys
import cv2
from cv2.bgsegm import *
import io

# image_shape = (224, 224, 3)
# camera = picamera.PiCamera(resolution=image_shape[:2])
# camera.exposure_mode = 'sports'

# stream = BytesIO()

# with picamera.PiCamera() as camera:
#     camera.resolution = (640, 480)
#     camera.framerate = 30
# 	camera.start_recording(stream, format='h264', quality=23)
#     camera.wait_recording(10)
#     camera.stop_recording()
import numpy as np

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# bg = cv2.createBackgroundSubtractorGMG()
bg = createBackgroundSubtractorMOG()
def subtract_bg(img, i):
	fgmask = bg.apply(img)#[:,:,np.newaxis] # adding the RGB channel
	cv2.imwrite("fgmask-{}.jpg".format(i), fgmask)
	fg_img = img.copy()
	fg_img[fgmask == 0] = [0,0,0]
	# img_bg = img - fgmask[:,:,np.newaxis]
	# return img - img_bg
	return fg_img
	# return bg.apply(img)

cap = cv2.VideoCapture(0)
i = 0


with picamera.PiCamera() as camera:
	camera.resolution = (640, 480)
	camera.framerate = 8
	stream = io.BytesIO()

	while True:
		camera.capture(stream, format="jpeg", use_video_port=True)
		frame = np.fromstring(stream.getvalue(), dtype=np.uint8)
		print(i)
		stream.seek(0)
		frame = cv2.imdecode(frame, 1)
		fg = subtract_bg(frame, i)
		cv2.imwrite("fg-{}.jpg".format(i), fg)
		i += 1

# while(True):
#     ret, frame = cap.read()
    # fg = subtract_bg(frame)
    # cv2.imwrite("fg-{}.jpg".format(i), fg)
    # i += 1

	
cap.release()
cv2.destroyAllWindows()


