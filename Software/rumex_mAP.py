from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.utils import compute_ap_range
from mrcnn.utils import compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# class that defines and loads the rumex dataset
class RumexDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "rumex")
		# define data locations
		images_dir = '../' + dataset_dir + '/Images/'
		annotations_dir = '../' + dataset_dir + '/Annotations/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip all images after 405 if we are building the train set
			if is_train and int(image_id) >= 1042: #1042 is the 405th image
				continue
			# skip all images before 405 if we are building the test/val set
			if not is_train and int(image_id) < 1042: #1042 is the 405th image
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
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
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('rumex'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "rumex_cfg"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 256
	DETECTION_NMS_THRESHOLD = 0.3
	DETECTION_MIN_CONFIDENCE = 0.7
	POST_NMS_ROIS_INFERENCE = 300
	DETECTION_MAX_INSTANCES = 15


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	AP5s = list()
	AP75s = list()
	ARs = list()
	APRs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP5, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.5)
		AP75, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.75)
		APr = compute_ap_range(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], verbose = 0)
		AR, _ = compute_recall(r["rois"], gt_bbox, iou=0.5) 
		# store
		APRs.append(APr)
		AP5s.append(AP5)
		AP75s.append(AP75)
		ARs.append(AR)
	# calculate the mean AP across all images
	mAP5 = mean(AP5s)
	mAP75 = mean(AP75s)
	mAR = mean(ARs)
	mAPr = mean(APRs)
	return mAP5, mAP75, mAR, mAPr

# load the train dataset
train_set = RumexDataset()
train_set.load_dataset('Dataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
# load the test dataset
test_set = RumexDataset()
test_set.load_dataset('Dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

# create config
cfg = PredictionConfig()
cfg.display()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_rumex_cfg_0005.h5', by_name=True)

# evaluate model on training dataset
train_mAP5, train_mAP75, train_mAR, train_mAPr = evaluate_model(train_set, model, cfg)
f_score_train = (2 * train_mAP5 * train_mAR)/(train_mAP5 + train_mAR)
print("Train mAP5: %.4f" % train_mAP5)
print("Train mAP75: %.4f" % train_mAP75)
print("Train mAPRange: %.4f" % train_mAPr)
print("Train mAR: %.4f" % train_mAR)
print("Train F1: %.4f" % f_score_train)

# evaluate model on test dataset
test_mAP5, test_mAP75, test_mAR, test_mAPr = evaluate_model(test_set, model, cfg)
f_score_test = (2 * test_mAP5 * test_mAR)/(test_mAP5 + test_mAR)
print("Test mAP: %.4f" % test_mAP5)
print("Test mAP75: %.4f" % test_mAP75)
print("Test mAPRange: %.4f" % test_mAPr)
print("Test mAR: %.4f" % test_mAR)
print("Test F1: %.4f" % f_score_test)