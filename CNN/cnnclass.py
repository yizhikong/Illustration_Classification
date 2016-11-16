import os
import sys
import time
import theano
import random
import cPickle
import imgDict
import numpy as np
import theano.tensor as T
from PIL import Image
from conv_layers import *
from collections import defaultdict, OrderedDict


def createData(path):
	imgDict.loadImgs(path)
	badIllust, goodIllust = imgDict.slip(imgDict.imgs)
	begin, addn = 0, 10020
	while True:
		illusts = badIllust[begin:begin+addn] + goodIllust[begin:begin+addn]
		if len(illusts) == 0:
			break
		random.shuffle(illusts)
		data_x, data_y, count = [], [], 0
		for illust in illusts:
			print count
			count += 1
			# illust = ('53361958', {'score': 501, 'filename': 'D:\\code\\commicDownload\\range\\53361958_3814_51_501.jpg'})
			try:
				img = Image.open(illust[1]['filename']).resize(image_shape)
				n_img = np.array(img, dtype='int8')
			except:
				print illust[1]['filename']
				continue
			# gray image
			if len(n_img.shape) == 2:
				continue
			# do not need the fourth channel
			n_img = n_img[:, :, :3]
			if n_img.shape != (image_shape[0], image_shape[1], 3):
				print n_img.shape
				continue
			x = np.rollaxis(n_img, 2)
			data_x.append(x)
			data_y.append(1 if illust[1]['score'] > 500 else 0)
		data_x, data_y = np.array(data_x, dtype='int8'), np.array(data_y, dtype='int8')
		cPickle.dump([data_x, data_y], open("dataset" + str(begin) + ".pkl", "w"))
		begin += addn

def loadData(image_shape=(100,100)):
	if os.path.exists("clearRGBdataset0.pkl"):
		datasets = []
		filenames = os.listdir(os.getcwd())
		for filename in filenames:
			if "clearRGB" in filename:
				print 'loading ' + filename + '...'
				datasets.append(cPickle.load(open(filename)))
				break
		data_x, data_y = datasets[0][0], datasets[0][1]
		del datasets[0]
		while len(datasets) != 0:
			data_x = np.vstack((data_x, datasets[0][0]))
			data_y = np.hstack((data_y, datasets[0][1]))
			del datasets[0]
		return [data_x, data_y]
	else:
		print 'begin create data...'
		createData('D:\\code\\commicDownload')
		return loadData()

def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

class ConvolutionalNeuralNetwork(object):

	def __init__(self, dataset, batch_size=50, learning_rate=0.03, class_num = 2, feature_length = 500,
	 img_size=(100,100), filter_size=(5,5), pool_size=(2,2), nkerns=(3, 10, 5), max_epoch = 25):
		self.batch_size = batch_size
		self.dataset = dataset
		self.max_epoch = max_epoch
		self.name = str(learning_rate) + '_' + str(feature_length) + '_' + str(nkerns) + '_' + str(filter_size) + '.txt'
		self.name = self.name.replace(' ', '').replace(',', '-').replace('(', 'T').replace(')', 'T')
		print 'name : ' + self.name
		print 'totally ' + str(len(dataset[0])) + " images"
		self.feature_length = feature_length
		self.class_num = class_num
		# x is the variable for input images, shape as (batch_size, feature_maps_num, image_h, image_w)
		# for one RGB image, feature_maps_num is 3 for R, G and B
		self.x = T.tensor4('x')
		# y is the variable for images' labels, just a list
		self.y = T.ivector('y')
		rng = np.random.RandomState(3435)

		# construct convolution_pool layers
		# next_layer_input and next_image_size are changed in each loop
		next_layer_input = T.cast(self.x.flatten(), dtype = "float64").reshape((batch_size, 3, img_size[0], img_size[1]))
		next_image_size = img_size
		# self.conv_layers save the convolutional layers of this network
		self.conv_layers = []
		feature_matches = zip(nkerns, nkerns[1:])
		for match in feature_matches:
			layer = LeNetConvPoolLayer(
				rng,
				input = next_layer_input,
				image_shape = (batch_size, match[0], next_image_size[0], next_image_size[1]),
				filter_shape = (match[1], match[0], filter_size[0], filter_size[1]),
				poolsize = pool_size,
				)
			self.conv_layers.append(layer)
			# the before layer's output is the input of next layer
			next_layer_input = layer.output
			# the image size is changed after twice filting(convolution and pooling)
			next_image_size = ((next_image_size[0] - filter_size[0] + 1) / pool_size[0],
				(next_image_size[1] - filter_size[1] + 1) / pool_size[1]
				)

		# construct hidden layer, its input is the output of the last convolution_pool layer
		# full connection here, n_in = feature_maps_num * img_pixel_nums
		hidden_layer_input = self.conv_layers[-1].output.flatten(2)
		self.hidden_layer = HiddenLayer(
			rng,
			input = hidden_layer_input,
			n_in = nkerns[-1] * next_image_size[0] * next_image_size[1],
			n_out = self.feature_length,
			activation = T.tanh
			)

		# the last layer, contain soft max
		self.classifier = LogisticRegression(input = self.hidden_layer.output,
			n_in = self.feature_length,
			n_out = self.class_num
			)
		self.cost = self.classifier.negative_log_likelihood(self.y)

		params = self.classifier.params + self.hidden_layer.params
		conv_layers_count = len(self.conv_layers)
		for i in range(conv_layers_count):
			params += self.conv_layers[conv_layers_count - i - 1].params
		if len(params) is not (conv_layers_count + 2) * 2 - 1:
			print "Error!!"
		# create a list of gradients for all model parameters
		self.grads = T.grad(self.cost, params)
		# create the updates list by automatically looping over all (params[i], grads[i]) pairs.
		self.updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, self.grads)]

		self.constructFuncs()
		self.train()

	def constructFuncs(self):
		dataset = self.dataset
		batch_size = self.batch_size
		index = T.lscalar()
		total = len(dataset[0])
		dataset[0] = np.array(dataset[0], dtype='float64')
		#split data
		train_set_x, train_set_y = dataset[0][:int(0.9*total)], dataset[1][:int(0.9*total)]
		valid_set_x, valid_set_y = dataset[0][int(0.9*total):], dataset[1][int(0.9*total):]
		# count batch num
		self.train_batch_num = len(train_set_x) / self.batch_size
		self.valid_batch_num = len(valid_set_x) / self.batch_size
		# share
		self.train_set_x, self.train_set_y = shared_dataset((train_set_x, train_set_y))
		self.valid_set_x, self.valid_set_y = shared_dataset((valid_set_x, valid_set_y))
		# for train and update
		self.train_model = theano.function(
			[index],
			self.cost,
			updates = self.updates,
			givens = {
				self.x : self.train_set_x[index*batch_size : (index+1)*batch_size],
				self.y : self.train_set_y[index*batch_size : (index+1)*batch_size]
				}
			)
		# for testing training data
		self.train_test_model = theano.function(
			[index],
			self.classifier.errors(self.y),
			givens={
				self.x: self.train_set_x[index * batch_size: (index + 1) * batch_size],
				self.y: self.train_set_y[index * batch_size: (index + 1) * batch_size]
				}
			)
		# for testing unknown data
		self.validate_model = theano.function(
			[index],
			self.classifier.errors(self.y),
			givens={
				self.x: self.valid_set_x[index * batch_size: (index + 1) * batch_size],
				self.y: self.valid_set_y[index * batch_size: (index + 1) * batch_size]
				}
			)

	def train(self):
		f = open(self.name, 'w')
		print '... training'
		epoch = 0
		best_val_perf = 0
		val_perf = 0
		test_perf = 0	   
		cost_epoch = 0	
		while (epoch < self.max_epoch):
			start_time = time.time()
			epoch = epoch + 1
			for index in xrange(self.train_batch_num):
				cost_epoch = self.train_model(index)
			train_losses = [self.train_test_model(i) for i in xrange(self.train_batch_num)]
			train_perf = 1 - np.mean(train_losses)
			val_losses = [self.validate_model(i) for i in xrange(self.valid_batch_num)]
			val_perf = 1- np.mean(val_losses)						
			print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
			f.write('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
			if val_perf >= best_val_perf:
				best_val_perf = val_perf
			if epoch == 10 and best_val_perf < 0.67:
				break
		print best_val_perf
		f.close()

if __name__ == '__main__':
	# data is narry
	dataset = loadData()
	cnn = ConvolutionalNeuralNetwork(dataset=dataset, learning_rate=0.03, filter_size = (5, 5), nkerns=(3, 5, 10), feature_length = 500, max_epoch = 20)
