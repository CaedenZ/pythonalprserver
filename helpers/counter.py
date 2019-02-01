import threading
import time


class counter:
    def __init__(self, max):
        self.num = 0
        self.max = max
        self.exceeded = False
        self.stopped = False

    def count(self):
        if(self.num <= self.max):
            self.num += 1
        elif not self.exceeded:  # If we have not set self.exceeded to True yet
            self.exceeded = True

    def reset(self):
        self.num = 0
        self.exceeded = False

    def resettime(self):
        self.num = 1
