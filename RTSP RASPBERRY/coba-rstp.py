#!/usr/bin/python3
import cv2
import time
import datetime
import io
#import rtsp
#from picamera2 import Picamera2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from flask import Flask, render_template, Response
import utils
import requests
import telepot
chat_id = -4962996227
bot = telepot.Bot('7972098389:AAGyJnB1JJPkVI8QHd8_28tYCndJyrVdwMM')

screenshot="none"
snaptime = True
def snapshot(im):
    cv2.imwrite("/home/pi/images/"+site+".jpg", im)
    bot.sendPhoto(chat_id, photo = open('/home/pi/images/'+site+'.jpg', 'rb'))
    return screenshot

app = Flask(__name__)

url1 = 'http://172.2.234.157:3000/send_text/?phone=6282297330080@c.us&message=ALERT_COK'
RepC9 = 'http://172.2.234.157:3000/send_text/?phone=6281328892000-1609742774@g.us&message=ALERT_NOTIF_AI_C9'
RepB5 = 'http://172.2.234.157:3000/send_text/?phone=6281328892000-1609742774@g.us&message=ALERT_NOTIF_AI_B5'
SUMOHAI = 'http://172.2.234.157:3000/send_text/?phone=6281328892000-1609742774@g.us&message=ALERT_NOTIF_AI_SUMOHAI'
RepB4 = 'http://172.2.234.157:3000/send_text/?phone=6281328892000-1609742774@g.us&message=ALERT_NOTIF_AI_B4'
myobj = {'untuklog': '32'}

######################################REP B5####################################################
def gen():
    model='efficientdet_lite0.tflite'
    num_threads=1

    dispW=1280
    dispH=720
    
    webCam=('rtsp://admin:Nvrtimika@2025@172.25.12.106:554/Streaming/Channels/102')
    webCam2='/home/pi/Not Respond.webp'
    cam=cv2.VideoCapture(webCam)
    cam2=cv2.VideoCapture(webCam2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    if not cam.isOpened():
    #print("failed to get stream...")
        cam=cv2.VideoCapture(webCam2)
    
    count_limit=30
    pos=(20,60)
    font=cv2.FONT_HERSHEY_SIMPLEX
    height=1.5
    weight=3
    myColor=(255,0,0)
    screenshot="none"
    snaptime = True
    def snapshot(im):
        cv2.imwrite("/home/pi/images/"+site+".jpg", im)
        bot.sendPhoto(chat_id, photo = open('/home/pi/images/'+site+'.jpg', 'rb'))
        return screenshot
    boxColor=(255,0,0)
    boxWeight=2
    t0=time.time()
    labelHeight=1.5
    labelColor=(0,255,0)
    labelWeight=(2)
    count=0
    
    fps=0

    base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
    detection_options=processor.DetectionOptions(max_results=3, score_threshold=.6)
    options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
    detector=vision.ObjectDetector.create_from_options(options)
    tStart=time.time()
    while cam.isOpened():
        now = datetime.datetime.now().time()
        start_time = datetime.time(7, 00, 0)
        end_time = datetime.time(17, 0, 0)
        success, im = cam.read()
    
        if not success:
            im = cam2.read()
            break
        imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        detections=detector.detect(imTensor)
        for detection in detections.detections:
            UL=(detection.bounding_box.origin_x,detection.bounding_box.origin_y)
            LR=(detection.bounding_box.origin_x+detection.bounding_box.width,detection.bounding_box.origin_y+detection.bounding_box.height)
            objName=detection.categories[0].category_name
            if objName=='person':
                im=cv2.rectangle(im,UL,LR,boxColor,boxWeight)
                cv2.putText(im,objName,UL,cv2.FONT_HERSHEY_PLAIN,labelHeight,labelColor,labelWeight)
                objname=detection.categories[0].category_name
                if start_time <= now <= end_time:
                    if snaptime== True:
                        site="cctv1"
                        screenshoot=snapshot(im)
                        snaptime = False
        ret, jpeg = cv2.imencode('.jpg', im)
        im=jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')
        cv2.destroyAllWindows()
####################################################REP B5########################################

#############################################REP c11##########################################
def gen1():
    model='efficientdet_lite0.tflite'
    num_threads=3

    dispW=1280
    dispH=720    
    webCam=('rtsp://admin:Nvrtimika@2025@172.25.12.106:554/Streaming/Channels/202')
    webCam2='/home/pi/Not Respond1.webp'
    cam=cv2.VideoCapture(webCam)
    cam2=cv2.VideoCapture(webCam2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
    #print("failed to get stream...")
        cam=cv2.VideoCapture(webCam2)
    
    pos=(20,60)
    font=cv2.FONT_HERSHEY_SIMPLEX
    height=1.5
    weight=3
    myColor=(255,0,0)
    screenshot="none"
    snaptime = True
    def snapshot(im):
        cv2.imwrite("/home/pi/images/"+site+".jpg", im)
        bot.sendPhoto(chat_id, photo = open('/home/pi/images/'+site+'.jpg', 'rb'))
        return screenshot
    boxColor=(255,0,0)
    boxWeight=2
    t0=time.time()
    labelHeight=1.5
    labelColor=(0,255,0)
    labelWeight=(2)
    count=0
    fps=0

    base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
    detection_options=processor.DetectionOptions(max_results=3, score_threshold=.7)
    options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
    detector=vision.ObjectDetector.create_from_options(options)
    tStart=time.time()
    while cam.isOpened():
        now = datetime.datetime.now().time()
        start_time = datetime.time(7, 00, 0)
        end_time = datetime.time(17, 0, 0)
        success, im = cam.read()
    
        if not success:
            im = cam2.read()
            break
        imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        detections=detector.detect(imTensor)
        for detection in detections.detections:
            UL=(detection.bounding_box.origin_x,detection.bounding_box.origin_y)
            LR=(detection.bounding_box.origin_x+detection.bounding_box.width,detection.bounding_box.origin_y+detection.bounding_box.height)
            objName=detection.categories[0].category_name
            if objName=='person':
                im=cv2.rectangle(im,UL,LR,boxColor,boxWeight)
                cv2.putText(im,objName,UL,cv2.FONT_HERSHEY_PLAIN,labelHeight,labelColor,labelWeight)
                objname=detection.categories[0].category_name
                if start_time <= now <= end_time:
                    if snaptime == True:
                        site="cctv2"
                        screenshoot=snapshot(im)
                        snaptime = False
    #                     time.sleep(2)
        ret, jpeg = cv2.imencode('.jpg', im)
        im=jpeg.tobytes()
        yield (b'--frame1\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')
        cv2.destroyAllWindows()
######################################rep c11###########################################
    
######################################SUMOHAI###########################################
def gen2():
    model='efficientdet_lite0.tflite'
    num_threads=1

    dispW=1280
    dispH=720
    
    webCam=('rtsp://admin:Nvrtimika@2025@172.25.12.106:554/Streaming/Channels/302')
    webCam2='/home/pi/Not Respond2.webp'
    cam=cv2.VideoCapture(webCam)
    cam2=cv2.VideoCapture(webCam2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
    #print("failed to get stream...")
        cam=cv2.VideoCapture(webCam2)

    pos=(20,60)
    font=cv2.FONT_HERSHEY_SIMPLEX
    height=1.5
    weight=3
    myColor=(255,0,0)
    screenshot="none"
    snaptime = True
    def snapshot(im):
        cv2.imwrite("/home/pi/images/"+site+".jpg", im)
        bot.sendPhoto(chat_id, photo = open('/home/pi/images/'+site+'.jpg', 'rb'))
        return screenshot
    boxColor=(255,0,0)
    boxWeight=2
    t0=time.time()
    labelHeight=1.5
    labelColor=(0,255,0)
    labelWeight=(2)
    count=0
    fps=0

    base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
    detection_options=processor.DetectionOptions(max_results=3, score_threshold=.8)
    options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
    detector=vision.ObjectDetector.create_from_options(options)
    tStart=time.time()
    while cam.isOpened():
        now = datetime.datetime.now().time()
        start_time = datetime.time(7, 00, 0)
        end_time = datetime.time(17, 0, 0)
        success, im = cam.read()
    
        if not success:
            im = cam2.read()
            break
        imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        detections=detector.detect(imTensor)
        for detection in detections.detections:
            UL=(detection.bounding_box.origin_x,detection.bounding_box.origin_y)
            LR=(detection.bounding_box.origin_x+detection.bounding_box.width,detection.bounding_box.origin_y+detection.bounding_box.height)
            objName=detection.categories[0].category_name
            if objName=='person':
                im=cv2.rectangle(im,UL,LR,boxColor,boxWeight)
                cv2.putText(im,objName,UL,cv2.FONT_HERSHEY_PLAIN,labelHeight,labelColor,labelWeight)
                objname=detection.categories[0].category_name
                if start_time <= now <= end_time:
                    if snaptime== True:
                        site="cctv3"
                        screenshoot=snapshot(im)
                        snaptime = False
        ret, jpeg = cv2.imencode('.jpg', im)
        im=jpeg.tobytes()
        yield (b'--frame2\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')
        cv2.destroyAllWindows()
######################################sumohai#################################################

######################################REP B4##############################################
def gen3():
    model='efficientdet_lite0.tflite'
    num_threads=1

    dispW=1280
    dispH=720
    
    webCam=('rtsp://admin:Nvrtimika@2025@172.25.12.106:554/Streaming/Channels/402')
    webCam2='/home/pi/Not Respond3.webp'
    cam=cv2.VideoCapture(webCam)
    cam2=cv2.VideoCapture(webCam2)
    cam=cv2.VideoCapture(webCam)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
    cam.set(cv2.CAP_PROP_FPS, 30)

    if not cam.isOpened():
    #print("failed to get stream...")
        cam=cv2.VideoCapture(webCam2)

    pos=(20,60)
    font=cv2.FONT_HERSHEY_SIMPLEX
    height=1.5
    weight=3
    myColor=(255,0,0)
    t0=time.time()
    boxColor=(255,0,0)
    boxWeight=2
    count=0
    labelHeight=1.5
    labelColor=(0,255,0)
    labelWeight=(2)
    screenshot="none"
    snaptime = True
    def snapshot(im):
        cv2.imwrite("/home/pi/images/"+site+".jpg", im)
        bot.sendPhoto(chat_id, photo = open('/home/pi/images/'+site+'.jpg', 'rb'))
        return screenshot
    fps=0

    base_options=core.BaseOptions(file_name=model,use_coral=False, num_threads=num_threads)
    detection_options=processor.DetectionOptions(max_results=4, score_threshold=.7)
    options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
    detector=vision.ObjectDetector.create_from_options(options)
    tStart=time.time()
    while cam.isOpened():
        now = datetime.datetime.now().time()
        start_time = datetime.time(7, 00, 0)
        end_time = datetime.time(17, 0, 0)
        success, im = cam.read()
    
        if not success:
            im = cam2.read()
            break
        imRGB=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        detections=detector.detect(imTensor)
        for detection in detections.detections:
            UL=(detection.bounding_box.origin_x,detection.bounding_box.origin_y)
            LR=(detection.bounding_box.origin_x+detection.bounding_box.width,detection.bounding_box.origin_y+detection.bounding_box.height)
            objName=detection.categories[0].category_name
            if objName=='person':
                im=cv2.rectangle(im,UL,LR,boxColor,boxWeight)
                cv2.putText(im,objName,UL,cv2.FONT_HERSHEY_PLAIN,labelHeight,labelColor,labelWeight)
                objname=detection.categories[0].category_name
                if start_time <= now <= end_time:
                     if snaptime== True:
                        site="cctv3"
                        screenshoot=snapshot(im)
                        snaptime = False
        ret, jpeg = cv2.imencode('.jpg', im)
        im=jpeg.tobytes()
        yield (b'--frame3\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + im + b'\r\n')  
        cv2.destroyAllWindows()
######################################REP B4#################################################

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame1')
# 
@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame2')
# 
@app.route('/video_feed3')
def video_feed3():
    return Response(gen3(),
                    mimetype='multipart/x-mixed-replace; boundary=frame3')
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index1.html')

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port =5002, debug=True, threaded=True)




