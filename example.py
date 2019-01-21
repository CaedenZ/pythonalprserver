import datetime
import os
import sys
import threading
import time
import cv2
import numpy as np
import pytesseract
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util


class VideoDetect:

    def __init__(self):
        self.stopped = False
        self.config = ('-l ukplate2 --oem 3 --psm 1')
        # sys.path.append("..")

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
        category_index = label_map_util.create_category_index(categories)

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if(not frames.empty()):
                outputFrames.put(self.detect_objects(frames.get()))

    def stop(self):
        self.stopped = True
        self.sess.close()

    def detect_objects(self, image_np):
        global i
        global config
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]

        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name(
            'detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name(
            'num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=.7,
            line_thickness=2)

        displayNoOfPlates = 0
        if(num_detections[0] >= 1):
            # print(boxes[0][0])
            for box in boxes[0]:
                if (box[0] > 0 and image_np.size != 0):
                    (ymin, xmin, ymax, xmax) = (box[0], box[1], box[2], box[3])
                    im_height, im_width, channels = image_np.shape
                    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                  ymin * im_height, ymax * im_height)
                    # print(left)
                    # print(right)
                    # print(top)
                    # print(bottom)

                    ###Clean image for better ocr###
                    # Crop the image
                    cropped_image = image_np[int(top):int(
                        bottom), int(left):int(right)].copy()

                    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

                    lower = np.array([0, 0, 100])
                    upper = np.array([255, 50, 255])

                    mask = cv2.inRange(hsv, lower, upper)

                    # iv = np.invert(mask)
                    # Convert image to gray
                    # cropped_image = cv2.cvtColor(
                    #     cropped_image, cv2.COLOR_BGR2GRAY)
                    # cropped_image = np.invert(cropped_image)
                    # cv2.imwrite('./crop/' + str(i) + '.png', cropped_image)
                    # # Increase image size for OCR
                    # cropped_image = cv2.resize(
                    #     mask, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    # # Denoise image
                    # filtered_image = cv2.adaptiveThreshold(cropped_image.astype(
                    #     np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 43, 20)
                    # kernel = np.ones((1, 1), np.uint8)
                    # opening = cv2.morphologyEx(
                    #     filtered_image, cv2.MORPH_OPEN, kernel)
                    # closing = cv2.morphologyEx(
                    #     opening, cv2.MORPH_CLOSE, kernel)
                    # # Smoothing
                    # ret1, th1 = cv2.threshold(
                    #     cropped_image, 180, 255, cv2.THRESH_BINARY)
                    # ret2, th2 = cv2.threshold(
                    #     th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # blur = cv2.GaussianBlur(th2, (1, 1), 0)
                    # ret3, th3 = cv2.threshold(
                    #     blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # #cropped_image = th3
                    # #Combine
                    # or_image = cv2.bitwise_or(cropped_image, closing)
                    text = pytesseract.image_to_string(
                        mask, config=self.config)
                    # save cropped image to output directory
                    # cv2.imwrite('./crop/' + str(i) + '.png', filtered_image)
                    # cv2.imshow("image",iv)
                    # if cv2.waitKey(25) == ord("q"):
                    #     self.stopped = True
                    displayNoOfPlates = num_detections[0]

        cv2.rectangle(image_np, (0, 0), (350, 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(image_np, "No. License Plates = " + str(num_detections[0]), (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # print(counter)
        # if(len(counting_mode) == 0):
        #     cv2.putText(image_np, "...", (10, 35), font, 0.8,
        #                 (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
        # else:
        #     cv2.putText(image_np, counting_mode, (10, 35), font,
        #                 0.8, (0, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
        return image_np
