"""
    Date: 12.07.2018
    Maintainer: Ahmet Kalay

    Gets xml files and images from specified folders.
    Gets coordinates from xml files and shows on the image as opencv window.
    Can bu used as checker for created xml files.
"""

from xml.etree import ElementTree as ET
import cv2
import numpy as np
import os

pathxml = "C:/Users/ahmet/Desktop/testxml/"
pathimage = "C:/Users/ahmet/Desktop/testimage/"
xmls = os.listdir(pathxml)
images = os.listdir(pathimage)
for file,image in  zip(xmls,images):
    tree = ET.parse(pathxml+file)
    doc = tree.getroot()
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
    #img = cv2.imread(pathimage+file.split(".")[0]+".jpg")
    img = cv2.imread(pathimage+image)
    print(pathxml+file)
    print(pathimage+image)
    for o in doc.findall('object'):
        for b in o.findall("bndbox"):
            xmin.append(b.find('xmin'))
            ymin.append(b.find('ymin'))
            xmax.append(b.find('xmax'))
            ymax.append(b.find('ymax'))



    for xm, ym, xx,yx,color in zip(xmin, ymin,xmax,ymax,colors):
        img = cv2.rectangle(img, (int(xm.text),int(ym.text)), (int(xx.text),int(yx.text)), color, 5)
        print(xm.text,ym.text,xx.text,yx.text)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 640, 480)
    cv2.imshow('frame', img)
    cv2.waitKey(0)
