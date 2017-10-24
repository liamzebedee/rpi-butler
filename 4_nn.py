import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.applications.vgg16 import preprocess_input
import numpy as np

import picamera
from time import sleep

from timeit import default_timer as timer
import os



vgg = applications.VGG16(include_top=False, weights='imagenet')


# The horizontal resolution (width) is rounded up to the nearest multiple of 32 pixels, while the vertical resolution (height) is rounded up to the nearest multiple of 16 pixels
def get_rpi_camera_buffer_shape(image_shape):
	w = np.ceil(image_shape[0] / 32) * 32
	h = np.ceil(image_shape[1] / 16) * 16
	return (int(w), int(h), image_shape[2])

image_shape = (160, 160, 3)
classes = ['beer', 'face', 'phone', 'mug', 'book', 'speaker', 'knife', 'bottle', 'hand', 'fishtank']
# class_indices = {'laptop': 8, 'mug': 9, 'face': 4, 'phone': 10, 'beer': 1, 'book': 2, 'speaker': 11, 'knife': 7, 'bottle': 3, 'hand': 6, 'auxcord': 0, 'fishtank': 5}
# num_classes = len(class_indices.keys())

print("Building model...")
# from rnn import ResnetBuilder
# model = ResnetBuilder.build_resnet_34(input_shape=(3,160,160,), num_outputs=12)

model = keras.models.model_from_json('{"class_name": "Sequential", "keras_version": "2.0.8", "config": [{"class_name": "Conv3D", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "conv3d_10", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "data_format": "channels_last", "filters": 32, "padding": "valid", "strides": [1, 1, 1], "dilation_rate": [1, 1, 1], "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "batch_input_shape": [null, 3, 160, 160, 3], "use_bias": true, "activity_regularizer": null, "kernel_size": [1, 3, 3]}}, {"class_name": "BatchNormalization", "config": {"beta_constraint": null, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "name": "batch_normalization_10", "epsilon": 0.001, "trainable": true, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "scale": true, "axis": -1, "gamma_constraint": null, "gamma_regularizer": null, "beta_regularizer": null, "momentum": 0.99, "center": true}}, {"class_name": "Activation", "config": {"activation": "selu", "trainable": true, "name": "activation_13"}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_10", "trainable": true, "data_format": "channels_last", "pool_size": [1, 2, 2], "padding": "valid", "strides": [1, 2, 2]}}, {"class_name": "Conv3D", "config": {"kernel_constraint": null, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "conv3d_11", "bias_regularizer": null, "bias_constraint": null, "activation": "linear", "trainable": true, "data_format": "channels_last", "padding": "valid", "strides": [1, 1, 1], "dilation_rate": [1, 1, 1], "kernel_regularizer": null, "filters": 16, "bias_initializer": {"class_name": "Zeros", "config": {}}, "use_bias": true, "activity_regularizer": null, "kernel_size": [1, 3, 3]}}, {"class_name": "BatchNormalization", "config": {"beta_constraint": null, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "name": "batch_normalization_11", "epsilon": 0.001, "trainable": true, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "scale": true, "axis": -1, "gamma_constraint": null, "gamma_regularizer": null, "beta_regularizer": null, "momentum": 0.99, "center": true}}, {"class_name": "Activation", "config": {"activation": "selu", "trainable": true, "name": "activation_14"}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_11", "trainable": true, "data_format": "channels_last", "pool_size": [1, 2, 2], "padding": "valid", "strides": [1, 2, 2]}}, {"class_name": "Conv3D", "config": {"kernel_constraint": null, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "conv3d_12", "bias_regularizer": null, "bias_constraint": null, "activation": "linear", "trainable": true, "data_format": "channels_last", "padding": "valid", "strides": [1, 1, 1], "dilation_rate": [1, 1, 1], "kernel_regularizer": null, "filters": 8, "bias_initializer": {"class_name": "Zeros", "config": {}}, "use_bias": true, "activity_regularizer": null, "kernel_size": [1, 3, 3]}}, {"class_name": "BatchNormalization", "config": {"beta_constraint": null, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "name": "batch_normalization_12", "epsilon": 0.001, "trainable": true, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Zeros", "config": {}}, "scale": true, "axis": -1, "gamma_constraint": null, "gamma_regularizer": null, "beta_regularizer": null, "momentum": 0.99, "center": true}}, {"class_name": "Activation", "config": {"activation": "selu", "trainable": true, "name": "activation_15"}}, {"class_name": "MaxPooling3D", "config": {"name": "max_pooling3d_12", "trainable": true, "data_format": "channels_last", "pool_size": [1, 2, 2], "padding": "valid", "strides": [1, 2, 2]}}, {"class_name": "Flatten", "config": {"trainable": true, "name": "flatten_4"}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_7", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 20, "use_bias": true, "activity_regularizer": null}}, {"class_name": "Activation", "config": {"activation": "selu", "trainable": true, "name": "activation_16"}}, {"class_name": "Dropout", "config": {"rate": 0.5, "trainable": true, "name": "dropout_4"}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "VarianceScaling", "config": {"distribution": "uniform", "scale": 1.0, "seed": null, "mode": "fan_avg"}}, "name": "dense_8", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "activation": "softmax", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 10, "use_bias": true, "activity_regularizer": null}}], "backend": "tensorflow"}')
	

print("Loading model...")
model.load_weights('model6.h5')

# model.compile(optimizer='sgd',
# 			  loss='categorical_crossentropy', 
# 			  metrics=['accuracy', 'top_k_categorical_accuracy'])

def classify(imgs):
	t_start = timer()

	# img = img/255.
	# img = preprocess_input(img)
	# x = np.expand_dims(img, 0)
	# x = vgg.predict(np.expand_dims(img, 0))
	# preds = model.predict(x)[0]

	# pred = np.argmax(preds)

	# for clx,inx in class_indices.items():
	# 	print("{} ({}%)".format(clx, preds[inx] / 10.))

	# print(preds)
	# index_to_class = dict(zip(class_indices.values(), class_indices.keys()))

	pred = model.predict(np.expand_dims(imgs, 0))
	if np.max(pred) > 0.3:
		stuff = {}
		for idx, val in enumerate(pred[0]):
			cat = classes[idx]
			stuff[cat] = val
		
		srt_stuff = sorted(stuff, key=stuff.get, reverse=True)
		
		for idx, label in enumerate(srt_stuff):
			if idx > 3:
				print("")
				return
			print("{} ({}%)".format(label, stuff[label]*100))

	t_end = timer()
	# print("{} ({}s)".format(index_to_class[pred], t_end - t_start))
	


camera = picamera.PiCamera(resolution=image_shape[0:2])
camera.exposure_mode = 'sports'

rpi_cam_buf_shape = get_rpi_camera_buffer_shape(image_shape)

last_capture = np.empty(np.prod(rpi_cam_buf_shape), dtype=np.uint8)

imgs = []

print("Running classify loop-")
try:
	# while input() != 'q':
	while True:
		for i in range(3):
			camera.capture(last_capture, 'rgb')
			img = last_capture.reshape(rpi_cam_buf_shape)[:image_shape[0], :image_shape[1]]
			if len(imgs) == 3:
				classify(imgs)
				imgs = []
			imgs.append(img)

except KeyboardInterrupt:
	pass

