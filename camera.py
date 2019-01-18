
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.
# # Imports


# In[21]:

import datetime
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import tensorflow as tf
import configparser


class camera:
    def __init__(self, src=0):
        self.IP = src
        self.stream = cv2.VideoCapture()
        self.stream.open("rtsp://" + src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.fourcc = cv2.VideoWriter_fourcc(*'X264')
        config = configparser.ConfigParser()
        config.read('server.ini')
        self.capture_duration = int(config['DEFAULT']['capture_duration'])
        self.fps = int(config['DEFAULT']['fps'])

    def start(self):
        threading.Thread(target=self.get, args=()).start()
        return self

    def resume(self):
        self.stopped = False

    def get(self):
        self.out = cv2.VideoWriter('/home/ubuntu/Documents/clip/' + str(int(time.time())) + '.avi', self.fourcc , self.fps, (1920,1080))
        self.start_time = time.time()
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()
                if((int(time.time()) - self.start_time) > self.capture_duration):
                    self.out.release()
                    self.start_time = time.time()
                    self.out = cv2.VideoWriter('/home/ubuntu/Documents/clip/' + str(int(time.time())) + '.avi', self.fourcc , self.fps, (1920,1080))
                self.out.write(self.frame)


    def stop(self):
        self.out.release
        self.stopped = True

    def destroy(self):
        self.out.release
        self.stopped = True
        self.stream.release
