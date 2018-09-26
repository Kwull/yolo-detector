from ctypes import *
import random
import os
import requests
import time
import urllib
import cv2
from skimage import draw
import numpy as np
import json
import datetime
import paho.mqtt.client as paho
import yaml
import pytz

with open("/config/detect.conf", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

netMain = None
metaMain = None
altNames = None

lib = CDLL("./darknet.so", RTLD_GLOBAL)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    custom_image_bgr = image
    custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(net), lib.network_height(net)),
                              interpolation=cv2.INTER_LINEAR)

    im, arr = array_to_image(custom_image)  # you should comment line below: free_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0,
                             pnum, 0)  # OpenCV
    # dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    free_detections(dets, num)
    return res


def initYolo(configPath ="./cfg/yolov3.cfg", weightPath ="yolov3.weights", metaPath="./cfg/coco.data"):
    global metaMain, netMain, altNames #pylint: disable=W0603
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `"+os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `"+os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `"+os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass


def performDetect(videoPath="test.mp4", taggedVideo="test.avi", thresh=0.25, storeTaggedVideo=False, storeKeyDetectionImages=False):
    global everyFrame  # pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"

    if not os.path.exists(videoPath):
        raise ValueError("Invalid image path `" + os.path.abspath(videoPath) + "`")

    video = cv2.VideoCapture(videoPath)
    _, frame = video.read()
    height, width, _ = frame.shape
    fps = round(video.get(cv2.CAP_PROP_FPS))

    if storeTaggedVideo:
        if os.name == "nt":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc(*'MP42')

        videoWriter = cv2.VideoWriter(taggedVideo, fourcc, fps, (width, height))

    tags = {} # resulted tags - maybe depricated soon
    objects = [] # resulted objects - tag + coordinates + image only for highest percentage

    currentFps = 0
    currentFrame = 0
    detections = ()

    while video.isOpened():
        currentFrame = currentFrame + 1
        currentFps = currentFps + 1
        if currentFrame > cfg['yolo']['tagEveryFrame']:
            shouldDetect = True
            currentFrame = 0
        else:
            shouldDetect = False

        _, frame = video.read()
        if frame is None:
            break

        if shouldDetect:
            detections = detect(netMain, metaMain, frame, thresh)
            for detection in detections:
                if (detection[0] not in tags) or (tags[detection[0]] < detection[1]):
                    tags[detection[0]] = np.rint(100 * detection[1])

        try:
            image = frame

            if shouldDetect:
                imcaption = []
                for detection in detections:
                    label = detection[0]
                    confidence = detection[1]
                    pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
                    imcaption.append(pstring)
                    # print(pstring)
                    bounds = detection[2]
                    shape = image.shape
                    yExtent = int(bounds[3])
                    xEntent = int(bounds[2])
                    # Coordinates are around the center
                    xCoord = int(bounds[0] - bounds[2] / 2)
                    yCoord = int(bounds[1] - bounds[3] / 2)
                    boundingBox = [
                        [xCoord, yCoord],
                        [xCoord, yCoord + yExtent],
                        [xCoord + xEntent, yCoord + yExtent],
                        [xCoord + xEntent, yCoord]
                    ]

                    imageUrl = taggedVideo.replace(".avi", "-"+detection[0]+".jpg")

                    tracked = {
                        "probability": np.rint(100 * detection[1]),
                        "frame": currentFps,
                        "x": xCoord,
                        "y": yCoord,
                        "x2": xCoord + xEntent,
                        "y2": yCoord + yExtent
                        }

                    objIsFound = False
                    isBestProbability = False
                    for obj in objects:
                        if (obj["name"] == detection[0]): 
                            objIsFound = True
                            obj["tracked"].append(tracked)
                            if (obj["probability"] < detection[1]):
                                isBestProbability = True
                                obj["probability"] = np.rint(100 * detection[1])
                                         
                    if (not objIsFound):
                        isBestProbability = True
                        objects.append({
                            "name": detection[0],
                            "probability": np.rint(100 * detection[1]),
                            "tracked": [tracked],
                            "image": imageUrl,
                            "fps": fps
                        })

                    if (storeTaggedVideo or storeKeyDetectionImages):
                        # Wiggle it around to make a 3px border
                        rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] for x in boundingBox],
                                                        shape=shape)
                        rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in boundingBox], [x[0] for x in boundingBox],
                                                        shape=shape)
                        rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in boundingBox], [x[0] for x in boundingBox],
                                                        shape=shape)
                        rr4, cc4 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] + 1 for x in boundingBox],
                                                        shape=shape)
                        rr5, cc5 = draw.polygon_perimeter([x[1] for x in boundingBox], [x[0] - 1 for x in boundingBox],
                                                        shape=shape)
                        boxColor = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                        draw.set_color(image, (rr, cc), boxColor, alpha=0.8)
                        draw.set_color(image, (rr2, cc2), boxColor, alpha=0.8)
                        draw.set_color(image, (rr3, cc3), boxColor, alpha=0.8)
                        draw.set_color(image, (rr4, cc4), boxColor, alpha=0.8)
                        draw.set_color(image, (rr5, cc5), boxColor, alpha=0.8)
                        cv2.putText(image, pstring, (xCoord, yCoord - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, boxColor, 2,
                                    cv2.LINE_AA)

                    if (storeKeyDetectionImages and isBestProbability):
                        cv2.imwrite(imageUrl, image)

            if (storeTaggedVideo):
                videoWriter.write(image)

        except Exception as e:
            print("Video processing error: " + str(e))

    if storeTaggedVideo:
        videoWriter.release()
    video.release()
    return tags, objects

# Real work started here

print 'Initializing yolo...'
initYolo(cfg['yolo']['configPath'], cfg['yolo']['weightPath'], cfg['yolo']['metaPath'])

print 'Connecting to mqtt broker...'
mqttClient = paho.Client('yolo')
mqttClient.username_pw_set(cfg['mqtt']['user'], cfg['mqtt']['password'])
mqttClient.connect(cfg['mqtt']['broker'], cfg['mqtt']['port'])
mqttClient.loop_start()

print 'Initialized and waiting for motion...'

inProgressRecordings = []
startDate = int(round(time.time() * 1000))

timeZone = 'Europe/Minsk'  # TODO: get from unifi server
pst = pytz.timezone(timeZone)

while True:
    time.sleep(cfg['unifi']['nvrScanInterval'])

    resp = requests.get('{}/api/2.0/recording?cause[]=motionRecording&startTime={}&sortBy=startTime&sort=asc&apiKey={}'
                        .format(cfg['unifi']['host'], startDate, cfg['unifi']['apiKey']))

    startDate = int(round(time.time() * 1000))

    if resp.status_code != 200:
        print ('Unifi Video API ERROR {}: {}'.format(resp.status_code, resp.text))
        continue

    recordings = resp.json()['data']
    print ('{} new motion recordings at Unifi Video'.format(recordings.__len__()))

    for inProgressRecording in inProgressRecordings:
        # re-fetch item from NVR to check status
        resp2 = requests.get('{}/api/2.0/recording/{}?apiKey={}'.format(cfg['unifi']['host'],
                                                                        inProgressRecording, cfg['unifi']['apiKey']))
        # todo - speed up by requesting all recordings by IDs
        updatedRecording = resp2.json()['data'][0]

        if not updatedRecording['inProgress']:
            recordings.insert(1, updatedRecording)
            inProgressRecordings.remove(inProgressRecording)

    for recording in recordings:
        recordingTime = pytz.utc.localize(datetime.datetime.fromtimestamp(recording['startTime']/1000))\
            .astimezone(pst).strftime('%Y-%m-%d %H:%M:%S')
        recordingStopTime = pytz.utc.localize(datetime.datetime.fromtimestamp(recording['endTime']/1000))\
            .astimezone(pst).strftime('%Y-%m-%d %H:%M:%S')

        print('{}: {} {} inProgress={}'.format(recordingTime, recording['meta']['cameraName'],
                                               recording['_id'], recording['inProgress']))

        if recording['inProgress']:
            inProgressRecordings.append(recording['_id'])
            print 'Skipping inProgress recording for now'
            continue

        recordingUrl = '{}/api/2.0/recording/{}/download?apiKey={}'.format(cfg['unifi']['host'],
                                                                           recording['_id'], cfg['unifi']['apiKey'])

        videoFile = urllib.urlretrieve(recordingUrl, '{}/{}-{}.mp4'.format(cfg['yolo']['motionFolder'],
                                                                           recording['_id'],
                                                                           recording['meta']['cameraName']))

        if os.stat(videoFile[0]).st_size < 1000:
            print ('Something is wrong with {}'.format(recording['_id']))
            continue

        filename, file_extension = os.path.splitext(os.path.basename(videoFile[0]))
        datePart = pytz.utc.localize(datetime.datetime.fromtimestamp(recording['startTime']/1000)).astimezone(pst).strftime('%Y/%m/%d')
        path = cfg['yolo']['processedFolder'] + '/' + datePart
        taggedVideo = path +'/' + filename + '.avi'

        if ((cfg['yolo']['storeTaggedVideo'] or cfg['yolo']['storeKeyDetectionImages']) and not os.path.exists(path)):
            os.makedirs(path)

        tags, objects = performDetect(videoFile[0], taggedVideo, cfg['yolo']['threshold'], cfg['yolo']['storeTaggedVideo'], cfg['yolo']['storeKeyDetectionImages'])

        detections = {
            'startTime': recordingTime,
            'endTime': recordingStopTime,
            'camera': recording['meta']['cameraName'],
            'recordingId': recording['_id'],
            'recordingUrl': recordingUrl,
            'tags': tags,
            "objects": objects
        }

        if cfg['yolo']['storeTaggedVideo']:
            detections['taggedVideo'] = taggedVideo

        if os.path.exists(videoFile[0]):
            os.remove(videoFile[0])

        jsonData = json.dumps(detections, indent=4, sort_keys=True)

        if len(detections['tags']) != 0:
            mqttTopic = cfg['mqtt']['rootTopic']+'/'+recording['meta']['cameraName'].lower()
            mqttClient.publish(mqttTopic, jsonData)
            print detections['recordingId'] + ' is published to mqtt ' + mqttTopic 
        else:    
            print detections['recordingId'] + ' nothing detected'
        # print(jsonData)

