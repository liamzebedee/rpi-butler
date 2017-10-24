'''Demonstrate Python wrapper of C apriltag library by running on camera frames.'''

from argparse import ArgumentParser
import cv2
import apriltag
import picamera
import picamera.array

import io
import numpy as np

from time import sleep

import subprocess
import os
import sys


cap = cv2.VideoCapture(0)

def talk_some_(shit):
	shit_cmd = 'pico2wave -l en-GB -w /var/local/pico2wave.wav "{}" | aplay'.format(shit)
	os.system(shit_cmd)

def do_the_thing(tag):
	try:
		name = tagids_to_objects[tag[1]]
		print(name)
		# if name == 'bottle':
		# 	talk_some_("anika, bitches be thirsty")
		# elif name == 'glasses':
		# 	# talk_some_("the final countdown")
		# 	talk_some_("anika, where the fook are me shades")
		# elif name == 'chair':
		# 	talk_some_("anika, sit down, be humble")
		# elif name == 'time2eat':
		# 	talk_some_("let me break this down for yall I'm hungry and I wanna make some sweeet sweet pasta")
		# sleep(2)
	except:
		print("try:", tag[1])

tagids_to_objects = {
	100:'bottle',
	97:'glasses',
	101:'chair',
	98:'time2eat'
}

def opencv_image_from(imgstream):
	GRAYSCALE = 0
	imgstream.seek(0)
	frame = np.fromstring(stream.getvalue(), dtype=np.uint8)
	gray = cv2.imdecode(frame, GRAYSCALE)
	return gray

def convert_img(img):
	img = cv2.cvtColor(stream.array, cv2.COLOR_RGB2LAB)
	img = img[:, :, 0]

	# img = cv2.cvtColor(stream.array, cv2.COLOR_RGB2GRAY)

	# img_conv = cv2.cvtColor(stream.array, cv2.COLOR_LAB2GRAY)
	# pritn(img_conv.shape)
	# img_conv = cv2.cvtColor(stream.array, cv2.COLOR_RGB2GRAY)
	return img


detector = apriltag.Detector(apriltag.DetectorOptions(nthreads=4, refine_edges=True, refine_decode=False, refine_pose=False, quad_blur=0.8, quad_decimate=1.0, ))

with picamera.PiCamera() as camera:
	# camera.resolution = (640, 480)
	camera.resolution = (2592, 1944)
	# camera.resolution = (1296, 972)

	# camera.resolution = (896, 768)
	# camera.exposure_mode = 'sports'

	camera.exposure_mode = 'antishake'
	# camera.exposure_compensation = -15

	camera.start_preview()
	print("Camera deets: {} {}, {}".format(camera.resolution, camera.framerate, camera.exposure_compensation))
	i = 0

	with picamera.array.PiRGBArray(camera) as stream:
		while True:
			camera.capture(stream, "rgb", use_video_port=True)

			sys.stdout.write(str(i) + "\t")
			i += 1
			
			img = convert_img(stream.array)
			

			stream.truncate(0)

			detections = []
			# detections, dimg = detector.detect(img, return_image=True)

			num_detections = len(detections)
			if num_detections == 0:
				print("")
				cv2.imwrite("stream4-{}.jpg".format(i), stream.array)
				continue
			else:
				# do_the_thing(detections[0])
				print('Detected {} tags - {}.\n'.format(num_detections, ', '.join(str(det[1]) for det in detections)))

				# for i, detection in enumerate(detections):
				# 	print('Detection {} of {}:'.format(i+1, num_detections))
				# 	print(detection.tostring(indent=2))
