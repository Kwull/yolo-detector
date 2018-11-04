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
            name_tag = meta.names[i]
        else:
            name_tag = altNames[i]
        res.append((name_tag, out[i]))
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
                    name_tag = meta.names[i]
                else:
                    name_tag = altNames[i]
                res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    free_detections(dets, num)
    return res


def init_yolo(config_path ="./cfg/yolov3.cfg", weight_path ="yolov3.weights", meta_path="./cfg/coco.data"):
    global metaMain, netMain, altNames

    if not os.path.exists(config_path):
        raise ValueError("Invalid config path `" + os.path.abspath(config_path) + "`")
    if not os.path.exists(weight_path):
        raise ValueError("Invalid weight path `" + os.path.abspath(weight_path) + "`")
    if not os.path.exists(meta_path):
        raise ValueError("Invalid data file path `" + os.path.abspath(meta_path) + "`")
    if netMain is None:
        netMain = load_net_custom(config_path.encode("ascii"), weight_path.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(meta_path.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(meta_path) as metaFH:
                meta_contents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", meta_contents, re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            names_list = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in names_list]
                except TypeError:
                    pass
        except Exception:
            pass


def perform_detect(video_path="test.mp4", tagged_video="test.avi", thresh=0.25, store_tagged_video=False,
                   store_key_detection_images=False):
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"

    if not os.path.exists(video_path):
        raise ValueError("Invalid image path `" + os.path.abspath(video_path) + "`")

    video = cv2.VideoCapture(video_path)
    _, frame = video.read()
    height, width, _ = frame.shape
    fps = round(video.get(cv2.CAP_PROP_FPS))

    if store_tagged_video:
        if os.name == "nt":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # fourcc = cv2.VideoWriter_fourcc(*'MP42')

        video_writer = cv2.VideoWriter(tagged_video, fourcc, fps, (width, height))

    tags = {}  # resulted tags - maybe depricated soon
    objects = []  # resulted objects - tag + coordinates + image only for highest percentage

    current_fps = 0
    current_frame = 0
    detections = ()

    while video.isOpened():
        current_frame = current_frame + 1
        current_fps = current_fps + 1
        if current_frame > cfg['yolo']['tagEveryFrame']:
            should_detect = True
            current_frame = 0
        else:
            should_detect = False

        _, frame = video.read()
        if frame is None:
            break

        if should_detect:
            detections = detect(netMain, metaMain, frame, thresh)
            for detection in detections:
                label = detection[0]
                confidence = np.rint(100 * detection[1])

                if (label not in tags) or (tags[label] < confidence):
                    tags[label] = confidence

        try:
            image = frame

            if should_detect:
                image_caption = []
                for detection in detections:
                    label = detection[0]
                    confidence = np.rint(100 * detection[1])

                    pstring = label + ": " + str(confidence) + "%"
                    image_caption.append(pstring)

                    bounds = detection[2]
                    shape = image.shape
                    y_extent = int(bounds[3])
                    x_entent = int(bounds[2])
                    # Coordinates are around the center
                    x_coord = int(bounds[0] - bounds[2] / 2)
                    y_coord = int(bounds[1] - bounds[3] / 2)
                    bounding_box = [
                        [x_coord, y_coord],
                        [x_coord, y_coord + y_extent],
                        [x_coord + x_entent, y_coord + y_extent],
                        [x_coord + x_entent, y_coord]
                    ]

                    image_url = tagged_video.replace(".avi", "-" + label + ".jpg")

                    tracked = {
                        "confidence": confidence,
                        "frame": current_fps,
                        "x": x_coord,
                        "y": y_coord,
                        "x2": x_coord + x_entent,
                        "y2": y_coord + y_extent
                        }

                    obj_is_found = False
                    is_best_confidence = False
                    for obj in objects:
                        if obj["name"] == label:
                            obj_is_found = True
                            obj["tracked"].append(tracked)
                            if obj["confidence"] < confidence:
                                is_best_confidence = True
                                obj["confidence"] = confidence

                    if not obj_is_found:
                        is_best_confidence = True
                        objects.append({
                            "name": label,
                            "confidence": confidence,
                            "tracked": [tracked],
                            "image": image_url,
                            "fps": fps
                        })

                    if store_tagged_video or store_key_detection_images:
                        # Wiggle it around to make a 3px border
                        rr, cc = draw.polygon_perimeter([x[1] for x in bounding_box], [x[0] for x in bounding_box]
                                                        , shape=shape)
                        rr2, cc2 = draw.polygon_perimeter([x[1] + 1 for x in bounding_box], [x[0] for x in bounding_box]
                                                          , shape=shape)
                        rr3, cc3 = draw.polygon_perimeter([x[1] - 1 for x in bounding_box], [x[0] for x in bounding_box]
                                                          , shape=shape)
                        rr4, cc4 = draw.polygon_perimeter([x[1] for x in bounding_box], [x[0] + 1 for x in bounding_box]
                                                          , shape=shape)
                        rr5, cc5 = draw.polygon_perimeter([x[1] for x in bounding_box], [x[0] - 1 for x in bounding_box]
                                                          , shape=shape)
                        box_color = (int(255 * (1 - (confidence ** 2))), int(255 * (confidence ** 2)), 0)
                        draw.set_color(image, (rr, cc), box_color, alpha=0.8)
                        draw.set_color(image, (rr2, cc2), box_color, alpha=0.8)
                        draw.set_color(image, (rr3, cc3), box_color, alpha=0.8)
                        draw.set_color(image, (rr4, cc4), box_color, alpha=0.8)
                        draw.set_color(image, (rr5, cc5), box_color, alpha=0.8)
                        cv2.putText(image, pstring, (x_coord, y_coord - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2,
                                    cv2.LINE_AA)

                    if store_key_detection_images and is_best_confidence:
                        cv2.imwrite(image_url, image)

            if store_tagged_video:
                video_writer.write(image)

        except Exception as e:
            print("Video processing error: " + str(e))

    if store_tagged_video:
        video_writer.release()
    video.release()
    return tags, objects

# Real work started here


print 'Initializing yolo...'
init_yolo(cfg['yolo']['configPath'], cfg['yolo']['weightPath'], cfg['yolo']['metaPath'])

print 'Connecting to mqtt broker...'
mqtt_client = paho.Client('yolo')
mqtt_client.username_pw_set(cfg['mqtt']['user'], cfg['mqtt']['password'])
mqtt_client.connect(cfg['mqtt']['broker'], cfg['mqtt']['port'])
mqtt_client.loop_start()

print 'Initialized and waiting for motion...'

in_progress_recordings = []

timeZone = 'Europe/Minsk'  # TODO: get from unifi server
pst = pytz.timezone(timeZone)

while True:
    start_date = (int(time.time()) - 1) * 1000
    time.sleep(cfg['unifi']['nvrScanInterval'])

    resp = requests.get('{}/api/2.0/recording?cause[]=motionRecording&startTime={}&sortBy=startTime&sort=asc&apiKey={}'
                        .format(cfg['unifi']['host'], start_date, cfg['unifi']['apiKey']))

    print ('{} - requesting new motion videos, startTime={}'
           .format(pytz.utc.localize(datetime.datetime.fromtimestamp(start_date / 1000))
                   .astimezone(pst).strftime('%Y-%m-%d %H:%M:%S'), start_date))

    # start_date = (int(time.time()) - 1) * 1000

    if resp.status_code != 200:
        print ('Unifi Video API ERROR {}: {}'.format(resp.status_code, resp.text))
        continue

    recordings = resp.json()['data']
    print ('{} new motion recordings at Unifi Video'.format(recordings.__len__()))

    for in_progress_recording in in_progress_recordings:
        # re-fetch item from NVR to check status
        resp2 = requests.get('{}/api/2.0/recording/{}?apiKey={}'.format(cfg['unifi']['host'],
                                                                        in_progress_recording, cfg['unifi']['apiKey']))
        # todo - speed up by requesting all recordings by IDs
        updated_recording = resp2.json()['data'][0]

        if not updated_recording['inProgress']:
            recordings.insert(1, updated_recording)
            in_progress_recordings.remove(in_progress_recording)

    for recording in recordings:
        recording_time = pytz.utc.localize(datetime.datetime.fromtimestamp(recording['startTime'] / 1000))\
            .astimezone(pst).strftime('%Y-%m-%d %H:%M:%S')
        recording_stop_time = pytz.utc.localize(datetime.datetime.fromtimestamp(recording['endTime'] / 1000))\
            .astimezone(pst).strftime('%Y-%m-%d %H:%M:%S')

        print('{}: {} {} inProgress={}'.format(recording_time, recording['meta']['cameraName'],
                                               recording['_id'], recording['inProgress']))

        if recording['inProgress']:
            in_progress_recordings.append(recording['_id'])
            print 'Skipping inProgress recording for now'
            continue

        recording_url = '{}/api/2.0/recording/{}/download?apiKey={}'.format(cfg['unifi']['host'],
                                                                            recording['_id'], cfg['unifi']['apiKey'])

        video_file = urllib.urlretrieve(recording_url, '{}/{}-{}.mp4'.format(cfg['yolo']['motionFolder'],
                                                                             recording['_id'],
                                                                             recording['meta']['cameraName']))

        if os.stat(video_file[0]).st_size < 1000:
            print ('Something is wrong with {}'.format(recording['_id']))
            continue

        filename, file_extension = os.path.splitext(os.path.basename(video_file[0]))
        date_part = pytz.utc.localize(datetime.datetime.fromtimestamp(recording['startTime'] / 1000))\
            .astimezone(pst).strftime('%Y/%m/%d')

        path = cfg['yolo']['processedFolder'] + '/' + date_part
        tagged_video = path + '/' + filename + '.avi'

        if (cfg['yolo']['storeTaggedVideo'] or cfg['yolo']['storeKeyDetectionImages']) and not os.path.exists(path):
            os.makedirs(path)

        tags, objects = perform_detect(video_file[0], tagged_video, cfg['yolo']['threshold'],
                                       cfg['yolo']['storeTaggedVideo'], cfg['yolo']['storeKeyDetectionImages'])

        detections = {
            'startTime': recording_time,
            'endTime': recording_stop_time,
            'camera': recording['meta']['cameraName'],
            'recordingId': recording['_id'],
            'recordingUrl': recording_url,
            'tags': tags,
            'objects': objects
        }

        if cfg['yolo']['storeTaggedVideo']:
            detections['taggedVideo'] = tagged_video

        if os.path.exists(video_file[0]):
            os.remove(video_file[0])

        json_data = json.dumps(detections, indent=4, sort_keys=True)

        if len(detections['tags']) != 0:
            mqtt_topic = cfg['mqtt']['rootTopic'] + '/' + recording['meta']['cameraName'].lower()
            mqtt_client.publish(mqtt_topic, json_data)
            print detections['recordingId'] + ' is published to mqtt ' + mqtt_topic
        else:    
            print detections['recordingId'] + ' nothing detected'
        # print(jsonData)

