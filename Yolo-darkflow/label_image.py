"""
    Date: 12.07.2018
    Maintainer: Ahmet Kalay

    Anaylzes images in the specified paths.
    Creates annotation files if image got a result greater than treshold otherwise deletes the image.
    CAUTION : Set options values according to your model before running.
              Set model to your configure file.
              Set load to number of last .profile file in ckpt folder.
"""
import cv2
from PIL._imaging import path

from darkflow.net.build import TFNet
import numpy as np
import time
import os
from lxml import etree
import xml.etree.cElementTree as ET

pathimage = "C:/Users/ahmet/Desktop/testimage/"
pathxml = "C:/Users/ahmet/Desktop/testxml/"
object_list = "tree"
tl_list = set()
br_list = set()

def read(tl_list,br_list):
    options = {
        # 'model': 'cfg/yolo.cfg',
        # 'load': 'bin/yolov2.weights',
        'model': 'cfg/tiny-yolo-voc-1c.cfg',
        'load': 1375,
        'threshold': 0.4,
        'gpu': 0.5
    }

    tfnet = TFNet(options)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
    #pathpre = "C:/Users/ahmet/Desktop/modeleddset/mobilelight/street lights/"
    images = os.listdir(pathimage)

    for i in images:
            try:
                image = cv2.imread(pathimage+i)
                #cv2.imshow('image',image)
                results = tfnet.return_predict(image)
                if results:
                    for color, result in zip(colors, results):
                                tl = (result['topleft']['x'], result['topleft']['y'])
                                br = (result['bottomright']['x'], result['bottomright']['y'])
                                label = result['label']
                                confidence = result['confidence']
                                tl_list.add(tl)
                                br_list.add(br)
                    write_xml(i,image,object_list,tl_list,br_list)
                    print("xml created",confidence)
                else :
                    print(pathimage+i,"no results")
                    os.remove(pathimage+i)
            except Exception as e:
                print(e,"asd")
                print(pathimage+i)
                os.remove(pathimage+i)
def write_xml(i,image,object_list,tl_list,br_list):
    """
    Creates annotation files that have same structure as Pascal VOC xml files.
    When using with another algorithms structure may need to be changed.

    :param folder: Image folder
    :param img: Image
    :param objects: Objects in the images
    :param tl: Coordinates of top left corner
    :param br: Corrdinates of bottom right corner
    :param savedir: Save directory
    :return: None
    """
    height, width, depth = image.shape
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = pathimage
    ET.SubElement(annotation, 'filename').text = i
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    for obj, topl, botr in zip(object_list, tl_list, br_list):
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = obj
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(topl[0])
        ET.SubElement(bbox, 'ymin').text = str(topl[1])
        ET.SubElement(bbox, 'xmax').text = str(botr[0])
        ET.SubElement(bbox, 'ymax').text = str(botr[1])

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)

    i = i.split(".")[0]+"."+i.split(".")[1].lower()
    if "jpg" in i:
        save_path = os.path.join(pathxml, i.replace('jpg', 'xml'))
    elif "png" in i:
        save_path = os.path.join(pathxml, i.replace('png', 'xml'))
    elif "jpeg" in i:
        save_path = os.path.join(pathxml, i.replace('jpeg', 'xml'))

    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)
    tl_list.clear()
    br_list.clear()
if __name__ == "__main__":
    read(tl_list,br_list)

