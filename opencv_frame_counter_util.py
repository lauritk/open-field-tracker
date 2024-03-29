#!/usr/bin/env python

import numpy as np
import cv2

video = "./test/TIM18SDM018_ACAC_1_alkuosa.avi"

video_capture = cv2.VideoCapture(video)
video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
while(True):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        count += 1

print(video_length, count)
# When everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
