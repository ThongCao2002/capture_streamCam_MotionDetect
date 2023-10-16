#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) Rau Systemberatung GmbH (rausys.de)
# MIT License
# credits: https://pyimagesearch.com/start-here/

import argparse
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import imutils
import time
import cv2

from settings import STREAM_URL, CAMERA_WIDTH, MIN_AREA, MAX_AREA, DEBUG, \
    REFERENCE_RELEVANT, RELEVANT_DEBOUNCE, OUTPUT_BACKLOG, \
    OUTPUT_INTERVAL, OUTPUT_PATH, OUTPUT_STATIC, NAME_CAM

argparser = argparse.ArgumentParser()
argparser.add_argument(
    '-s', '--stream',
    help='Stream URL (RTSP) to get video feed from',
    nargs='?', type=str,
    default=STREAM_URL
)
argparser.add_argument(
    '-w', '--window',
    help='Show relevant feeds as X11 window',
    action='store_true')
argparser.add_argument(
    '--debug',
    help='Output more information',
    action='store_true'
)
args = argparser.parse_args()
DEBUG = args.debug or DEBUG

# Helper variables:
# last static saving tracking
last_static = datetime.now()
# allow for more consistent "continuous relevant event" handling
debounce_counter = 0


@dataclass
class ReferenceFrames:
    """ Helper class to manage frames """
    frame: object = None
    timestamp: datetime = field(init=False)
    previous: list = field(default_factory=lambda: [])
    latest_capture: object = None

    def set_frame(self, frame: object):
        """ Sets reference frame which is used to calculate difference
        from the current camera image """
        if self.frame is None or self.timestamp <= datetime.now() - timedelta(minutes=REFERENCE_RELEVANT):
            self._set_frame(frame=frame)

    def _set_frame(self, frame: object):
        if DEBUG: print('Updating reference frame')
        self.frame = frame
        self.timestamp = datetime.now()

    def append(self, frame: object, contour: int, contour_index: int, contour_amount: int):
        # Improvement idea: Constant rolling buffer - as soon as occupied=True
        self.previous.append([frame, contour])

        if DEBUG:
            print(f'[{contour_index+1}/{contour_amount}] {contour}')
            self.save_image(frame=frame, contour=contour)

   

def get_stream():
    if not args.stream:
        print('Stream URI for RTSP server not specified! Exiting')
        exit(1)
    return cv2.VideoCapture(args.stream)


if __name__ == '__main__':
    print('Initializing stream...')
    frames = ReferenceFrames()
    vs = get_stream()

    while True:
        # grab the current frame and initialize the occupied/unoccupied
        retrieved, full_frame = vs.read()
        if not retrieved:
            print('Error retrieving image from stream; reinitializing')
            vs = get_stream()
            continue

        if full_frame is None: continue
        occupied = False

        # resize the frame, convert it to grayscale, and blur it
        scaled_frame = imutils.resize(full_frame, width=CAMERA_WIDTH)
        y, x, channels = scaled_frame.shape
        #frame = full_frame[:, START_CROP_X:x]
        frame = scaled_frame.copy()

        # src_cropped = src[top_margin:src.shape[0], :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if frames.frame is None:
            frames.set_frame(frame=gray)
            continue

        # compute the absolute difference between the current frame and first frame
        frameDelta = cv2.absdiff(frames.frame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # if the contour is too small, ignore it
        relevant_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
        contour_sizes = [cv2.contourArea(c) for c in relevant_contours]

        for i, (contour, contour_size) in enumerate(zip(relevant_contours, contour_sizes)):
            # reset reference picture; this is to help detect if there's actual motion
            # if multiple consecutive pictures change, it's likely we are dealing with motion
            frames._set_frame(frame=gray)
            # compute the bounding box for the contour, draw it on the frame,
            # and update the status
            (x, y, w, h) = cv2.boundingRect(contour)
            # x = x + START_CROP_X  # ensure relative boxes are rendered properly
            cv2.rectangle(scaled_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            debounce_counter = RELEVANT_DEBOUNCE
            occupied = True

        # save image to output static image every two seconds
        if OUTPUT_STATIC and last_static < datetime.now() - timedelta(seconds=OUTPUT_INTERVAL):
            last_static = datetime.now()
            file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
            print(f'Saving image to backlog: {file_name} {NAME_CAM}')
            backlog_path = os.path.join(OUTPUT_PATH, NAME_CAM)
            os.makedirs(backlog_path, exist_ok=True)
            cv2.imwrite(os.path.join(backlog_path, file_name), scaled_frame)

   
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    vs.release()  # vs.stop()
    cv2.destroyAllWindows()
