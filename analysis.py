import configparser
import datetime
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from helpers.counter import counter
from helpers.enum_models import Model
from mongo.mongodb import mongodb
from utils import label_map_util
from utils import visualization_utils as vis_util


class analysis:
    def __init__(self, camera='tmp', model="", location="unkown", minx=0, maxx=0, miny=0, maxy=0, interestArea=False):
        self.cameraName = camera
        self.thresholdcheck = counter(75)
        self.outputFrames = deque()
        self.inputFrames = deque()
        self.write = False
        self.detectionmodel = model
        self.stopped = False
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.location = location
        self.mongo = mongodb()
        config = configparser.ConfigParser()
        config.read('server.ini')
        modeldir = config['DEFAULT']['modeldir']
        self.clipdir = config['DEFAULT']['clipdir']
        self.fps = int(config['DEFAULT']['fps'])
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.IA = interestArea
        # consPath = self.clipdir + self.cameraName + os.sep + self.detectionmodel + os.sep + 'const.mp4'
        # self.consout = cv2.VideoWriter(consPath, self.fourcc, self.fps, (1920, 1080))
        # What model to download.
        MODEL_NAME=modeldir + self.detectionmodel
        # MODEL_NAME = '/home/ubuntu/Downloads/s                                    sd_mobilenet_v2_coco_2018_03_29'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT=MODEL_NAME + '/frozen_inference_graph_face.pb'

        print('loading model:' + PATH_TO_CKPT)
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS=os.path.join(
            MODEL_NAME, 'labelmap.pbtxt')

        NUM_CLASSES=1

        self.detection_graph=tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def=tf.GraphDef()

            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph=fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name = '')

        self.sess=tf.Session(graph = self.detection_graph)

        label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
        categories=label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
        self.category_index=label_map_util.create_category_index(categories)

    def start(self):
        threading.Thread(target = self.analyse, args = ()).start()
        return self

    def analyse(self):
        while not self.stopped:
            if self.inputFrames:
                # print(len(self.inputFrames))
                frame=self.inputFrames.popleft()
                if frame is not None:
                    self.detect_objects(frame)

        self.out.release()
        # self.consout.release()

    def stop(self):
        print('stopped')
        try:

            print('releasing')
        except:
            print('Video already released')

        self.stopped=True
        self.sess.close()

    def load_image_into_numpy_array(self, image):
        (im_width, im_height)=image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def detect_objects(self, image_np):
        # Do detection here
        # image_np = self.load_image_into_numpy_array(image)

        image_np_expanded=np.expand_dims(image_np, axis = 0)
        image_tensor=self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes=self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores=self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes=self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections=self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections)=self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict = {image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates = True,
            min_score_thresh = .5,
            line_thickness = 2)

        # self.consout.write(image_np)

        # print(num_detections[0])
        if(not self.IA):
            if(scores[0][0] >= 0.5):
                print(scores[0][0])
                if(self.thresholdcheck.num == 0):
                    # Start a new video
                    print('writing')
                    dirPath=self.clipdir + self.cameraName + os.sep + self.detectionmodel
                    if not os.path.exists(dirPath):
                        os.makedirs(dirPath)
                    filePath=dirPath + os.sep + str(int(time.time())) + '.mp4'
                    self.out=cv2.VideoWriter(
                        filePath, self.fourcc, self.fps, (1920, 1080))
                    log={"file": filePath, "time": str(
                        int(time.time())), "location": self.location}
                    self.mongo.insertone("log", log)
                self.out.write(image_np)
                self.thresholdcheck.resettime()

            else:
                if (self.thresholdcheck.exceeded):
                    self.out.release()
                    self.thresholdcheck.reset()
                else:
                    # Stops from falsely recording after finishing a recording
                    if(self.thresholdcheck.num != 0):
                        self.thresholdcheck.count()
                        self.out.write(image_np)
        else:
            if(scores[0][0] >= 0.5):
                [h, w]=image_np.shape[:2]
                result = []
                for s in scores[0]:
                    if (s > 0.5):
                        result.append(s)
                for i in range(len(result)):
                    box=boxes[0][i]
                    # Get the origin co-ordinates and the length and width till where the face extends
                    ymin, xmin, ymax, xmax=[v for v in box]

                    x=int(xmin * w)
                    y=int(ymin * h)
                    xm=int(xmax * w)
                    ym=int(ymax * h)

                    if(x < self.maxx and xm > self.minx and y < self.maxy and ym > self.miny):
                        print('detected')
                        print(scores[0][0])
                        if(self.thresholdcheck.num == 0):
                            # Start a new video
                            dirPath=self.clipdir + self.cameraName + os.sep + self.detectionmodel
                            if not os.path.exists(dirPath):
                                os.makedirs(dirPath)
                            filePath=dirPath + os.sep + \
                                str(int(time.time())) + '.mp4'
                            self.out=cv2.VideoWriter(
                                filePath, self.fourcc, self.fps, (1920, 1080))
                            log={"file": filePath, "time": str(
                                int(time.time())), "location": self.location}
                            self.mongo.insertone("log", log)
                        self.out.write(image_np)
                        self.thresholdcheck.resettime()

            else:
                if (self.thresholdcheck.exceeded):
                    self.out.release()
                    self.thresholdcheck.reset()
                else:
                    # Stops from falsely recording after finishing a recording
                    if(self.thresholdcheck.num != 0):
                        self.thresholdcheck.count()
                        self.out.write(image_np)
