class threadedtimer:
    def __init__(self, duration):
        self.start = 0
        self.duration = duration
        self.exceeded = False

    def start(self):
        threading.Thread(target=self.startTimer, args=()).start()
        return self

    def begin(self):
        if(start >= duration) {
            self.exceeded = True
        }
        start += 1

    def reset(self):
        self.start = 0
        self.exceeded = False


    def stop(self):
