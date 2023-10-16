#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) Rau Systemberatung GmbH (rausys.de)
# MIT License

import os
from distutils.util import strtobool


# URL to grab and analyze stream from
#STREAM_URL = os.environ['CAMERA_URL']
STREAM_URL = 'rtsp://admin:HQ-0907979696@anphukp5c.vncctv.com:1024/Streaming/Unicast/channels/401'
# Resize original picture to (if changed MIN_AREA+MAX_AREA & CROP_X should be changed too)
CAMERA_WIDTH = int(os.environ.get('CAMERA_WIDTH', 1080))
# Minutes the reference image is considered relevant;
# important for accurate results during day-night transition
REFERENCE_RELEVANT = int(os.environ.get('CAMERA_REFERENCE_REFRESH', 10))
# How many frames in order have to be irrelevant to consider
# a movement event finished
RELEVANT_DEBOUNCE = 20

# Movement detection
# MIN_AREA: how many pixels have to change in order to be considered movement
# MAX_AREA: maximum pixels changing - everything above is considered a change in scenery (e.g. light on during night),
#   which results in not being considered for being a valid candidate
MIN_AREA: int = int(os.environ.get('CAMERA_MIN_THRESHOLD', 12000))
MAX_AREA: int = int(os.environ.get('CAMERA_MAX_THRESHOLD', 115000))

# Periodically save static image of stream
OUTPUT_STATIC: bool = strtobool(os.environ.get('CAMERA_OUTPUT', 'True'))
OUTPUT_PATH: str = os.environ.get('CAMERA_OUTPUT_PATH', '.')
OUTPUT_INTERVAL: int = int(os.environ.get('CAMERA_OUTPUT_INTERVAL', 1))

# Save most relevant static image of captured motion events
OUTPUT_BACKLOG: bool = strtobool(os.environ.get('CAMERA_BACKLOG', 'True'))

DEBUG: bool = strtobool(os.environ.get('CAMERA_DEBUG', 'False'))
NAME_CAM = 'cam3'

