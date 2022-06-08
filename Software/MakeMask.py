from fileinput import filename
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from PIL import Image, ImageDraw


annotationDir = str("../Dataset/annotations/")
imageDir = str("../Dataset/images/train/")

def extract_info(filename):
    # load and parse the file
    tree = ET.parse(filename)
    # get the root of the document
    root = tree.getroot()
    file = str(root.find('filename').text)
    annotationNum = os.path.splitext(file)
    # extract each bounding box
    boxes = []
    for member in root.findall('object'):
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)
        boxes.append([xmin, ymin, xmax, ymax])
    # extract image dimensions
    width = int(root.find('size')[0].text)
    height = int(root.find('size')[1].text)
    print("Boxes ", boxes)
    print("Width ", width)
    print("Height ", height)
    print("Filename ", annotationNum[0])
    return annotationNum[0], boxes, width, height

for image in os.scandir(imageDir):
    imageName = os.fsdecode(image)
    if imageName.endswith(".jpg"):
        base = os.path.basename(image)
        split = os.path.splitext(base)
        imageNum = str(split[0])
        # print(imageNum)
        annotationNum, boxes, width, height = extract_info(annotationDir + imageNum +".xml")
        # creating new Image object
        img = Image.new('P',(width,height),'#00ff00')
        img1 = ImageDraw.Draw(img)
        # img1.rectangle((0,0,width,height), fill = 'white', outline='white')
        # img1.rectangle((0,0,width,height), fill = 'blue', outline='blue')
        for b in range(len(boxes)):
        # create rectangle image
            img1.rectangle((boxes[b]), fill ='red',outline='red')
        # img.show()
        mask = '../Dataset/masks/train/' + annotationNum + '.png'
        img.save(mask)
        continue
    else:
        continue
    

# for file in os.scandir(annotationDir):
#     filename = os.fsdecode(file)
#     if filename.endswith(".xml"):
#         annotationNum, boxes, width, height = extract_info(annotationDir + imageNum +'.xml')
#         # creating new Image object
#         img = Image.new('RGB',(width,height),0)
#         img1 = ImageDraw.Draw(img)
#         for b in range(len(boxes)):
#         # create rectangle image
#             img1.rectangle((boxes[b]), fill ='white',outline='white')
#         # img.show()
#         mask = '../Dataset/masks/' + annotationNum + '.png'
#         img.save(mask)
#         continue
#     else:
#         continue
