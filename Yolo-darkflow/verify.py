"""
    Date: 12.07.2018
    Maintainer: Ahmet Kalay

    Gets images from specified folder and runs model on them.
    Shows result on an opencv window.
    Press any key to run next image.
    CAUTION : Set options values according to your model before running.
"""

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os

def verify():
    options = {
        'model': 'cfg/tiny-yolo-voc-1c.cfg',
        'load': 1375,
        'threshold': 0.4,
        'gpu': 0.5
    }


    tfnet = TFNet(options)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

    #pathpre = "C:/Users/ahmet/Desktop/modeleddset/mobilelight/full/"
    pathpre = "C:/Users/ahmet/Desktop/testimage/"
    images = os.listdir(pathpre)
    c = set()
    for i in images:
            img = cv2.imread(pathpre+i)
            results = tfnet.return_predict(img)
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                img = cv2.rectangle(img, tl, br, color, 5)
                img = cv2.putText(img, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                c.add(confidence)
            print(pathpre+i,c)
            c.clear()
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 640, 480)
            cv2.imshow('frame', img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    verify()
