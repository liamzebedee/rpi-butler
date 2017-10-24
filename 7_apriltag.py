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


cap = cv2.VideoCapture(0)

# os.system("sudo ln -s /dev/stdout /var/local/pico2wave.wav")

def talk_some_(shit):
	shit_cmd = 'pico2wave -l en-GB -w /var/local/pico2wave.wav "{}" | aplay'.format(shit)
	os.system(shit_cmd)
	# process = subprocess.Popen(shit_cmd, shell=True, pipe_stdout=True)
	# process.wait()


# talk_some_("Alahu akbar")
# raise Exception("")

def do_the_thing(tag):
	try:
		name = tagids_to_objects[tag[1]]
		if name == 'bottle':
			talk_some_("anika, bitches be thirsty")
		elif name == 'glasses':
			# talk_some_("the final countdown")
			talk_some_("anika, where the fook are me shades")
		elif name == 'chair':
			talk_some_("anika, sit down, be humble")
		elif name == 'time2eat':
			talk_some_("let me break this down for yall I'm hungry and I wanna make some sweeet sweet pasta")
		sleep(2)
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


detector = apriltag.Detector(apriltag.DetectorOptions(nthreads=4, refine_decode=True, quad_blur=0.4))
with picamera.PiCamera() as camera:
	camera.resolution = (640, 480)
	camera.start_preview()
	# camera.framerate = 2
	# stream = io.BytesIO()
	i = 0

	with picamera.array.PiRGBArray(camera) as stream:
		while True:
			camera.capture(stream, "rgb", use_video_port=False)

			print(i)
			i += 1
			# gray = opencv_image_from(stream)
			# print(stream.array.shape)
			gray = cv2.cvtColor(stream.array, cv2.COLOR_RGB2GRAY)
			# print(gray.shape)
			stream.truncate(0)

			detections, dimg = detector.detect(gray, return_image=True)

			num_detections = len(detections)
			if num_detections == 0:
				continue
			else:
				# print(detections[0].id)
				do_the_thing(detections[0])

			# print('Detected {} tags.\n'.format(num_detections))

			# for i, detection in enumerate(detections):
			# 	print('Detection {} of {}:'.format(i+1, num_detections))
			# 	print(detection.tostring(indent=2))
