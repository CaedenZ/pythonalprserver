import sys
import io
import cv2
import base64
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from flask import (Flask, Response, make_response, render_template, request,
                   send_file, jsonify)
from flask_cors import CORS,cross_origin

from inference_image_api import image_api
from lpr_ocr_api import lpr_ocr_api
sys.path.append("../")

# from VideoCap import VideoCap


app = Flask(__name__)
cors = CORS(app, resources={r'/*': {"origins": '*'}})
app.config['CORS_HEADER'] = 'Content-Type'
# video_cap = VideoCap().start()
image_manager = image_api()
lpr_manager = lpr_ocr_api()
# def gen(cap):
#     while True:
#         frame = cap.getFrame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(video_cap),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/api/v1/postlpr/', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type'])
def receive_lpr():
    data = request.files['image']
    imgd = data.read()

    imagedata = np.fromstring(imgd,np.uint8)
    image = cv2.imdecode(imagedata,cv2.IMREAD_COLOR)
    img, text = lpr_manager.processImgfile(image)
    # imgplot = plt.imshow(image)
    # plt.show()
    # response = make_response(img.tobytes())
    # response = make_response(io.BytesIO(img))
    returnData = {"img": base64.b64encode(img), "text": text}
    response = make_response(jsonify(returnData))
    response.headers.set('Content-Type', 'application/json')
    return response



@app.route('/api/v1/lpr/', methods=['GET'])
def process_lpr():
    url = request.args.get('url')
    print('===========================')
    print(url)
    img = lpr_manager.run(url)

    # imgplot = plt.imshow(img)
    # plt.show()
    response = make_response(img.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')
    return response

@app.route('/api/v1/image/', methods=['GET'])
def process_img():
    # print('this is running')
    # print(request.form['image'])
    # url = request.form['image']
    url = request.args.get('url')
    print('===========================')
    print(url)
    img = image_manager.run(url)

    # imgplot = plt.imshow(img)
    # plt.show()
    response = make_response(img.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')
    return response


if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=True)
