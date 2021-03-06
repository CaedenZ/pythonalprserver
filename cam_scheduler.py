import threading
import time
import datetime

from camera import camera


class cam_scheduler:
    def __init__(self, cam, schedule_list = []):
        self.camera = cam
        self.sch_list = schedule_list
        self.stopped = False

    def start(self):
        threading.Thread(target=self.check_schedule, args=()).start()
        return self

    def check_schedule(self):
        while not self.stopped:
            isScheduled = False
            # Check if today is scheduled
            for s in self.sch_list:
                # 1 = Monday, 2 = Tues, etc.
                if (datetime.date.today().isoweekday() == s.day):
                    starttime = s.start_time
                    endtime = s.end_time
                    parsedstarttime = datetime.datetime.strptime(starttime, "%Y-%m-%dT%H:%M:%S.%fZ") + datetime.timedelta(hours=8)
                    parsedendtime = datetime.datetime.strptime(endtime, "%Y-%m-%dT%H:%M:%S.%fZ") + datetime.timedelta(hours=8)
                    if(datetime.date.today() >= parsedstarttime && datetime.date.today() <= parsedendtime):
                        isScheduled = True
                        break
                else:
                    isScheduled = False
            if (isScheduled):
                # If the camera is currently not running, we want to start it
                if (self.camera.stopped):
                    self.camera.start()
            else:
                # If the camera is not scheduled to run, we want to stop it
                self.camera.stop()
            sleep(600)

    def stop(self):
        self.stopped = True
