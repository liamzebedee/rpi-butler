import picamera
from time import sleep
import os
import sys

image_shape = (224, 224, 3)
camera = picamera.PiCamera(resolution=image_shape[:2])
camera.exposure_mode = 'sports'

categories = [
	'random',
	'speaker',
	'fishtank',
	'laptop',
	'face',
	'chair',
	'book',
	'waterbottle',
	'basil'
]


def rm_younger_than(_dir, x_seconds, current_date=None):
	cwdir = os.getcwd()
	os.chdir(_dir)

	print("Deleting younger than %ds" % x_seconds)
	if not current_date:
		current_date = int(os.popen("date +%s").read())
	files_n_dates = os.popen("stat --printf='%n %X\n' *.jpg").read().strip()
	for line in files_n_dates.split('\n'):
		file, date = line.partition(" ")[::2]
		date = int(date)
		diff = current_date - date
		if diff < x_seconds:
			print("{} - born {}s ago - now gone.".format(file, diff))
			os.system("rm "+file)
		else:
			print("{} - skipped".format(file))

	os.chdir(cwdir)

cat = ''

while True:
	try:
		if cat == '':
			cat = input("What category are we capturing? ")
			if cat == '':
				print("Nothing - ok, cya!")
				sys.exit(0)
		
		print("Capturing for category {}".format(cat))
		os.system("mkdir -p imgs/{}".format(cat))
		sleep(1)

		for i, filename in enumerate(camera.capture_continuous('imgs/'+cat+'/image{timestamp:%j-%H-%M-%S-%f}.jpg')):
			print(i)
			# i += 1

	except KeyboardInterrupt as e:
		current_date = int(os.popen("date +%s").read())
		ans = input("Continue capturing (y/n)? ")
		print("Deleting the past 2 seconds of shitty capturing")
		rm_younger_than('imgs/{}/'.format(cat), 3, current_date)
		
		if ans == 'y':
			continue
		else:
			print("Stopping capture of {}".format(cat))
			cat = ''
			continue

# for cat in categories:
# 	print("Your next object to capture is {}.".format(cat))
# 	print("Press [Enter] when ready, and you'll have 2 seconds to get in position")
# 	input()
# 	sleep(2)

	
	

# 	try:
			
# 	except KeyboardInterrupt:
# 		print("Stopping capture of {}".format(cat))
# 		print("Also deleting the past 3 seconds of shitty capturing")
# 		rm_younger_than('imgs/{}/'.format(cat), 3)
# 		continue