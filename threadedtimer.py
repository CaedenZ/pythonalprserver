import threading
import time


class threadedtimer:
    def __init__(self, duration):
        self.timeRan = 0
        self.duration = duration
        self.exceeded = False
        self.stopped = False

    def start(self):
        threading.Thread(target=self.startTimer, args=()).start()
        return self

    def startTimer(self):
        while not self.stopped:
            if(self.timeRan <= self.duration):
                self.timeRan += 1
                sleep(1)
            elif not self.exceeded:  # If we have not set self.exceeded to True yet
                self.exceeded = True

    def reset(self):
        self.timeRan = 0
        self.exceeded = False
        self.stopped = False

    def stop(self):
        self.stopped = True
