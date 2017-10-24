import picamera
from time import sleep
import os

camera = picamera.PiCamera()
camera.exposure_mode = 'sports'

try:
	for i, filename in enumerate(camera.capture_continuous('imgs-test/{counter:02d}.jpg')):
		print(filename)
		i += 1
		sleep(0.1)
except KeyboardInterrupt:
	pass