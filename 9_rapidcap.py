print("Booting up...")
import cv2
import io
import time
import threading
import picamera
import picamera.array
from PIL import Image
import signal
import sys

import apriltag
import numpy as np
import os
import queue
import signal

from scipy.misc import imfilter, imread
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2

# Create a pool of image processors
done = False
lock = threading.Lock()
pool = []

# w, h = (1296, 972)
# w,h= (720, 480)
w,h = (640, 480)
GRAY = 'L'



def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    os.system('mpc stop')
    done = True
    os._exit(-1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
# signal.signal(signal.SIGKILL, signal_handler)

def talk_some_(shit):
	shit_cmd = 'pico2wave -l en-GB -w /var/local/pico2wave.wav "{}" | aplay'.format(shit)
	os.system(shit_cmd)


"""
house	http://uk5.internet-radio.com:8318/live
soul 	http://109.74.196.76:8022/stream
exit radio uk 	http://178.79.158.160:8454/stream
"""
radio_stations = {
	124: 1,
	98: 2,
	120: 3,
	125: 4
}

def play_radio(id_):
	if id_ in radio_stations:
		os.system("mpc play {}".format(radio_stations[id_]))

class ImageProcessor(threading.Thread):
	def __init__(self):
		super(ImageProcessor, self).__init__()
		self.stream = io.BytesIO()
		self.event = threading.Event()
		self.terminated = False
		self.detector = apriltag.Detector(apriltag.DetectorOptions(nthreads=4, refine_edges=True, refine_decode=False, refine_pose=False, quad_blur=0.0, quad_decimate=1.0, ))

		self.start()

	def run(self):
		# This method runs in a separate thread
		global done

		while not self.terminated:
			if self.event.wait(1):
				try:
					self.stream.seek(0)

					# Read the image and do some processing on it
					img = Image.open(self.stream)
					frame = np.array(img.convert(GRAY))
					
					# sharpen
					# kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
					# frame = cv2.filter2D(frame, -1, kernel)

					# image = frame/255.
					# psf = np.ones((5, 5)) / 25
					# # image = conv2(image, psf, 'same')
					# # image += 0.1 * image.std() * np.random.standard_normal(image.shape)
					# deconvolved = restoration.wiener(image, psf, 1, clip=False)

					# deconvolved *= 255.
					# deconvolved = deconvolved.astype('uint8')
					# print(deconvolved.dtype)

					# frame = color.rgb2lab(frame)[:,:,1]
					# gray = cv2.imdecode(frame, cv2.COLOR_RGB2LAB)[:,:,0]

					# frame = np.fromstring(self.stream.getvalue(), dtype=np.uint8).reshape(w,h)
					# gray = cv2.imdecode(frame, cv2.COLOR_RGB2GRAY)

					# img = cv2.imdecode(np.fromstring(self.stream.getvalue(), dtype=np.uint8), cv2.COLOR_BGR2GRAY)
					# raise 0
					detections = self.detector.detect(frame, return_image=False)
					if len(detections) > 0:
						print(', '.join([ str(d[1]) for d in detections ]))
						play_radio(detections[0][1])
					# print(len(detections))

					# img.save("stream5-{}.jpg".format(time.time()))

					#...
					#...
					# Set done to True if you want the script to terminate
					# at some point
					#done=True
				except AssertionError:
					done=True
				finally:
					# Reset the stream and event
					self.stream.seek(0)
					self.stream.truncate()
					self.event.clear()
					# Return ourselves to the pool
					with lock:
						pool.append(self)

def streams():
	while not done:
		with lock:
			try:
				processor = pool.pop()
			except IndexError:
				continue
		yield processor.stream
		processor.event.set()


with picamera.PiCamera() as camera:
	pool = [ImageProcessor() for i in range (4)]
	camera.resolution = (w, h)
	# camera.exposure_mode = 'sports'
	camera.exposure_mode = 'antishake'
	# Set the framerate appropriately; too fast and the image processors
	# will stall the image pipeline and crash the script
	camera.framerate = 10

	print("Beginning capture")
	camera.start_preview()
	camera.capture_sequence(streams(), 'jpeg', use_video_port=True)

print("Capturing.")

# Shut down the processors in an orderly fashion
while pool:
	with lock:
		processor = pool.pop()
	processor.terminated = True
	processor.join()