import base64
import io
import sys

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, json, make_response, render_template,
                   request, send_file, send_from_directory)

from camera import camera
from cam_scheduler import cam_scheduler
from flask_cors import CORS, cross_origin

sys.path.append("../")

# from VideoCap import VideoCap


app = Flask(__name__)
cors = CORS(app, resources={r'/*': {"origins": '*'}})
app.config['CORS_HEADER'] = 'Content-Type'
threadDict = {}


@app.route('/api/v1/newcamera/', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def addcamera():
    data = request.get_json(force=True)
    # print(data)
    camDatainString = data.get('camera')
    camData = json.loads(camDatainString)
    IP = camData.get('camera_ip_address')
    location_id = camData.get('location_id')


    minx = camData.get('minx')
    maxx = camData.get('maxx')
    miny = camData.get('miny')
    maxy = camData.get('maxy')
    # print(camData.get('camera_ip_address'))
    # IP = data.get('ipAdd')
    moduleList = data.get('camModuleList')
    schedule_list = data.get('camSchedList')
    moduleIDlist = []
    for m in moduleList:
        moduleIDlist.append(m.get('cameraModuleIdentity').get('moduleid'))
    print(IP)
    print(location_id)
    print(moduleIDlist)
    IA = False
    if(maxx != 0):
        IA = True
    cam = camera(IP,moduleIDlist,location_id,minx,maxx,miny,maxy,IA)
    ######## activate without scheduler ########
    cam.start()
    threadDict[IP] = cam

    ######## activate with scheduler ########
    # cScheduler = cam_scheduler(cam, schedule_list)
    # cScheduler.start()
    # threadDict[IP] = cScheduler

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


@app.route('/api/v1/getframe/', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type'])
def getframe():
    data = request.get_json()
    IP = data.get('ipAdd')
    cam = threadDict.get(IP)
    frame = cam.frame
    returnData = {"img": base64.b64encode(frame)}
    response = make_response(jsonify(returnData))
    response.headers.set('Content-Type', 'application/json')
    return response


@app.route('/api/v1/removecamera/', methods=['DELETE'])
@cross_origin(origin='*', headers=['Content-Type'])
def removecamera():
    data = request.get_json()
    IP = data.get('ipAdd')
    cam = threadDict.get(IP)
    cam.destroy()
    threadDict.pop(IP)

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


@app.route('/api/v1/testlogs/', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type'])
def testLo():
    data = request.get_json()
    print(data.get('ipAdd'))
    IP = data.get('ipAdd')
    cam = camera(IP)
    cScheduler = cam_scheduler(cam, schedule_list)
    threadDict[cam.IP] = cScheduler

    return json.dumps({'success':True}), 200, {'ContentType':'application/json'}


@app.route('/api/v1/<file_name>', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type'])
def livestreamresponse(file_name):
    return send_from_directory('/media/ubuntu/storagedrive/models-master/research/object_detection/Server/livestream', file_name, cache_timeout=-1)


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)
