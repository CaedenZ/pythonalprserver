import base64
import io
import sys

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, make_response, render_template,
                   request, send_file, send_from_directory)

from camera import camera
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
    data = request.get_json()
    print(data.get('ipAdd'))
    IP = data.get('ipAdd')
    cam = camera(IP)
    cam.start()
    threadDict[IP] = cam
    response = make_response("You Are Stupiddd")
    response.headers.set('Content-Type', 'application/json')
    return response


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

    response = make_response("you are stupid")
    response.headers.set('Content-Type', 'application/json')
    return response


# @app.route('/api/v1/servestream/', methods=['GET'])
# @cross_origin(origin='*', headers=['Content-Type'])
# def livestream():
#     return send_from_directory('/media/ubuntu/storagedrive/models-master/research/object_detection/Server/livestream', 'test.m3u8')

@app.route('/api/v1/<identifier>/<file_name>', methods=['GET'])
@cross_origin(origin='*', headers=['Content-Type'])
def livestreamresponse(file_name, identifier):
    return send_from_directory('/media/ubuntu/storagedrive/models-master/research/object_detection/Server/livestream', file_name, cache_timeout=-1)


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)
