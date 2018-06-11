# -*- coding: utf-8 -*

import argparse
from PIL import Image
import os
import dlib
import cv2
from shutil import copyfile

def cropFace(file, outDir, detector, landmarks, size):
    middlePtNumber = 29
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    det = detector(img, 1)
    if (len(det) == 0):
        print "error file {}".format(file)
        copyfile(file, os.path.join("errorImg", file[file.rfind("/") + 1:]))
        return
    feature = det[0]
    part = landmarks(img, feature)
    middleY = part.part(middlePtNumber).y
    imgHeight = img.shape[1] # shape[1] is width, set height equal to width
    imgTop = middleY - imgHeight / 2
    imgBottom = middleY + imgHeight / 2
    if (imgTop < 0):
        imgTop = 0
        imgBottom = imgHeight
    if (imgBottom > img.shape[0]):
        imgTop = img.shape[0] - imgHeight
        imgBottom = img.shape[0]
    cropImg = img[imgTop:imgBottom, :, :]
    cropSave = Image.fromarray(cropImg.astype('uint8'))
    resizedImg = cropSave.resize((size, size))
    fullSavePath = os.path.join(outDir, file[file.rfind("/") + 1:])
    print "save {}".format(fullSavePath)
    resizedImg.save(fullSavePath)

parser = argparse.ArgumentParser()
parser.add_argument('imageFolder')
parser.add_argument('--outDir')
parser.add_argument('--outSize', default=128, type=int)
args = parser.parse_args()

path = args.imageFolder
counter = 0
if (not os.path.exists(args.outDir)):
    os.mkdir(args.outDir)
if (not os.path.exists(path)):
    print "{} note exists".format(path)
else:
    detector = dlib.get_frontal_face_detector()
    landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fileList = os.listdir(path)
    for file in fileList:
        if (os.path.isdir(file)):
            continue
        counter += 1
        if (counter%10 != 0):
            continue
        filePath = os.path.join(path, file)
        print "crop {}".format(filePath)
        cropFace(filePath, args.outDir, detector, landmarks, args.outSize)
        
