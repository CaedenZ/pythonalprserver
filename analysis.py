import datetime
import os
import queue
import sys
import threading
import time
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from utils import label_map_util
from utils import visualization_utils as vis_util
from helpers.enum_models import Model


class analysis:
    def __init__(self, model=Model.NONE):
        self.outputFrames = deque()
        self.inputFrames = deque()
        self.detectionmodel = model.value
        self.stopped = False
        # What model to download.
        MODEL_NAME = '/media/ubuntu/storagedrive/models-master/research/object_detection/model45'
        # MODEL_NAME = '/home/ubuntu/Downloads/ssd_mobilenet_v2_coco_2018_03_29'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join(
            '/media/ubuntu/storagedrive/models-master/research/object_detection/licenseplate/data', 'labelmap.pbtxt')

        NUM_CLASSES = 1

        self.detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=detection_graph)

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def analyse(self):
        while not self.stopped:
            if(self.inputFrames):
                self.outputFrames.append(self.detect_objects(self.inputFrames.popleft()))

    def stop(self):
        self.stopped = True
        self.sess.close()

    def detect_objects(self, image_np):
        #Do detection here
