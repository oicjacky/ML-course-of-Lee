'''
google keyword: use background subtraction before object detection

Ref: 
    1. [How to Use Background Subtraction Methods](https://docs.opencv.org/4.5.3/d1/dc5/tutorial_background_subtraction.html)
'''
from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=None)
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()


## [capture]
capture = cv.VideoCapture(0 if args.input is None else cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)


while True:
    ret, frame = capture.read()
    if frame is None:
        print(ret, frame.shape)
        pass
    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame)

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == ord('d') or keyboard == 27:
        break