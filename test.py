from analysis import analysis
from helpers.enum_models import Model
import cv2


src = 'rtsp://192.168.1.219'

stream = cv2.VideoCapture()
stream.open(src)


myAnalyseThread = analysis('test',Model(2).name,'yishun')
myAnalyseThread.start()
count = 0
while (count < 500):
    (grabbed, frame)= stream.read()
    myAnalyseThread.inputFrames.append(frame)
    count += 1


myAnalyseThread.stop()
