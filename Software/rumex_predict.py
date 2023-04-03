# detect rumex in photos with mask rcnn model
from fileinput import filename
import os
from os import listdir
from turtle import back
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib import colors, cm
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from mrcnn.utils import Dataset
from mrcnn.visualize import display_differences, display_instances, display_images
import time
import random

CLASS_NAMES = ['BG','Rumex']

# class that defines and loads the rumex dataset
class RumexDataset(Dataset):
	# load the dataset definitions
	def loadDataset(self, datasetDir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "Rumex")
		# define data locations
		images_dir = datasetDir + '/Images/'
		annotationDir = datasetDir + '/Annotations/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip all images after 405 if we are building the train set
			if is_train and int(image_id) >= 1042: #405th photo
				continue
			# skip all images before 405 if we are building the test/val set
			if not is_train and int(image_id) < 1042: #405th photo
				continue
			imagePath = images_dir + filename
			annotationPath = annotationDir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=imagePath, annotation=annotationPath)

	# extract bounding boxes from an annotation file
	def extractBoxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def loadMask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extractBoxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('Rumex'))
		return masks, asarray(class_ids, dtype='int32')
	

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path'], info['id']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "Rumex_cfg"
	# number of classes (background + rumex)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	DETECTION_NMS_THRESHOLD = 0.3
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
	# DETECTION_MIN_CONFIDENCE = 0.8
	# DETECTION_NMS_THRESHOLD = 0.3
	DETECTION_MIN_CONFIDENCE = 0.8
	POST_NMS_ROIS_INFERENCE = 300
	DETECTION_MAX_INSTANCES = 15

def speedTest(dataset, model, cfg, n_images):
	start = time.time()
	for i in range(n_images):
		image = dataset.load_image(i)
		scaled_image = mold_image(image, cfg)
		sample = expand_dims(scaled_image, 0)
		yhat = model.detect(sample, verbose=0)[0]
	elapsed = time.time() - start
	print("Prediction Latency: ", elapsed)	


# plot a number of photos with ground truth and predictions
def visualise(dataset, model, cfg, n_images=10):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.loadMask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		start = time.time()
		yhat = model.detect(sample, verbose=0)[0]
		elapsed = time.time() - start
		print("Detection latency:", elapsed)
		# define subplot
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		# pyplot.title('Actual')
		pyplot.axis('off')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		# pyplot.title('Predicted')
		pyplot.axis('off')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	pyplot.show()


def VisualiseMasks(dataset, model, cfg, n_images=1, is_train = True, save = False):
	if (is_train):
		path = '../Results/Masks/1/Train/'
	else:
		path = '../Results/Masks/1/Test/'	
	for i in range(n_images):
		image = dataset.load_image(i)
		mask, id = dataset.loadMask(i)
		_ , info = dataset.image_reference(i)
		filename = info
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		# start = time.time()
		r = model.detect(sample, verbose=0)[0]
		# elapsed = time.time() - start
		# print("Detection Latency: ", elapsed)
		pyplot.clf()
		pyplot.axis('off')
		ax = pyplot.gca()
		display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'],
								  figsize=(5, 5),
								  ax = ax)
		if (save):
			if not os.path.isdir("%s" % str(path)):
				os.makedirs("%s" % str(path))
			pyplot.savefig('%s%s.jpg' % (str(path),str(filename)),
					bbox_inches='tight',
					orientation= 'landscape')
		else:
			pyplot.show()								  
								  
		# mrcnn.visualize.display_differences(image = image,
		# 						gt_box = bbox,
		# 						gt_class_id = class_ids,
		# 						gt_mask = mask,
		# 						pred_box = r['rois'],
		# 						pred_class_id = r['class_ids'],
		# 						pred_score = r['scores'],
		# 						pred_mask = r['masks'],
		# 						class_names = CLASS_NAMES,
		# 						title="",
		# 						show_mask = True,
		# 						show_box = True)

def sideBySide(dataset, model, cfg, n_images=10, is_train = True, save = False):
	#Sets the output directory based upon dataset
	if (is_train):
		path = '../Results/BBox/1/Train/'
	else:
		path = '../Results/BBox/1/Test/'
	pyplot.figure(figsize=(5,5))
	pyplot.tight_layout()
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.loadMask(i)
		_ , info = dataset.image_reference(i)
		filename = info
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		start = time.time()
		yhat = model.detect(sample, verbose=0)[0]
		elapsed = time.time() - start
		print("Prediction latency:", elapsed)
		pyplot.clf()
		# define subplot
		pyplot.subplot(1, 2, 1)
		# plot raw pixel data
		pyplot.imshow(image)
		# pyplot.title('Actual')
		pyplot.axis('off')
		# plot masks
		cmap = cm.hsv
		weed = colors.colorConverter.to_rgba('r')
		for j in range(mask.shape[2]):
			gtOverlay = cmap(mask[:, :, j])
			gtOverlay[mask[:, :, j] <=0,:] = None
			gtOverlay[mask[:, :, j] >0,:] = weed
			pyplot.imshow(gtOverlay, alpha=0.5)
		# get the context for drawing boxes
		pyplot.subplot(1, 2, 2)
		# plot raw pixel data
		pyplot.imshow(image)
		# pyplot.title('Predicted')
		pyplot.axis('off')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
		if (save):
			if not os.path.isdir("%s" % str(path)):
				os.makedirs("%s" % str(path))
			pyplot.savefig('%s%s.jpg' % (str(path),str(filename)),
					bbox_inches='tight',
					orientation= 'landscape')
		else:
			pyplot.show()

def display_image(dataset, ind):
    pyplot.figure(figsize=(5,5))
    pyplot.imshow(dataset.load_image(ind))
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title('Original Image')
    pyplot.show()

# load the train dataset
train_set = RumexDataset()
train_set.loadDataset('rumex', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = RumexDataset()
test_set.loadDataset('rumex', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
cfg.display()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'mask_rcnn_rumex_cfg_0005.h5'
model.load_weights(model_path, by_name=True)
# plot predictions for train dataset
# visualise(train_set, model, cfg)
# VisualiseMasks(train_set, model, cfg, n_images=20)
# plot predictions for test dataset
# visualise(test_set, model, cfg)
VisualiseMasks(test_set, model, cfg, n_images=135, is_train = False, save = True)
# ind = 9
# display_image(test_set, ind)
# # predict_and_plot_differences(test_set, ind)
# sideBySide(test_set, model, cfg, 135, is_train = False, save = True)
# sideBySide(train_set, model, cfg, 405, is_train = True, save = True)
# speedTest(test_set, model, cfg,n_images=135)