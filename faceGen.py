import cv2
import os
import dlib
import numpy
import argparse
import MovingLSQ as MLSQ
import math
from PIL import Image

def getBound(img, part):
    xMin = len(img[0])
    xMax = 0
    yMin = len(img)
    yMax = 0
    for i in range(part.num_parts):
        if (part.part(i).x < xMin):
            xMin = part.part(i).x
        if (part.part(i).x > xMax):
            xMax = part.part(i).x
        if (part.part(i).y < yMin):
            yMin = part.part(i).y
        if (part.part(i).y > yMax):
            yMax = part.part(i).y
    return xMin, xMax, yMin, yMax

def setLandmark(part, xMin, yMin):
    # 特征点是从完整的原图中定位的，需要从原图中截取face部分，坐标需要调整
    ctrlPts = numpy.zeros((part.num_parts, 2))
    for i in range(part.num_parts):
        ctrlPts[i] = [part.part(i).x - xMin, part.part(i).y - yMin]
    return ctrlPts

def transOverMargin(transIdx, width, height):
    xMin = 0
    xMax = width
    yMin = 0
    yMax = height
    for i in range(len(transIdx)):
        if (transIdx[i][0] < xMin):
            xMin = transIdx[i][0]
        if (transIdx[i][1] < yMin):
            yMin = transIdx[i][1]
        if (transIdx[i][2] > xMax):
            xMax = transIdx[i][2]
        if (transIdxi[i][3] > yMax):
            yMax = transIdx[i][3]
    xOverRight = math.ceil(xMax - width)
    yOverBottom = math.ceil(yMax - height)
    xOverLeft = -math.floor(xMin)
    yOverTop = -math.floor(yMin)
    return xOverLeft, xOverRight, yOverTop, yOverBottom

def faceGen(file):
    print "processing {}".format(file)
    eigenPath = "eigen.jpg"
    eigenImg = cv2.cvtColor(cv2.imread(eigenPath), cv2.COLOR_BGR2RGB)
    fileImg = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

    eigenDetector = dlib.get_frontal_face_detector()
    detector = dlib.get_frontal_face_detector() 
    eigenDet = eigenDetector(eigenImg, 1)
    det = detector(fileImg, 1)
    eigenFeature = eigenDet[0]
    feature = det[0]

    landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    eigenPart = landmarks(eigenImg, eigenDetector)
    part = landmarks(fileImg, detector)
    if (eigenPart.num_parts != part.num_parts):
        print "landmarks part number not equal"
        return

    eigenXmin, eigenXmax, eigenYmin, eigenYmax = getBound(eigenImg, eigenPart)
    xmin, xmax, ymin, ymax = getBound(fileImg, part)

    eigenCtrlPts = setLandmark(eigenPart, eigenXmin, eigenYmin)
    ctrlPts = setLandmark(part, xMin, yMin)

    # moving from ctrlPts to eigenCtrlPts
    solver = MSLQ.MovingLSQ(ctrlPts, eigenCtrlPts)
    cropWidth = xMax - xMin
    cropHeight = yMax - yMin
    imgIdx = numpy.zeros((cropWidth * cropHeight, 2))
    for i in range(cropHeight):
        for j in range(cropWidth):
            imgIdx[i * (cropWidth) + j] = [j, i]
    transImgIdx = solver.Run_Rigid(imgIdx)
    transImgMap = transIdx.reshape((cropHeight, cropWidth, 2))
    leftMargin, rightMargin, topMargin, bottomMargin = transOverMargin(transImgIdx, cropWidth, cropHeight)

    transImg = numpy.zeros(cropHeight + int(topMargin) + int(bottomMargin),
                            cropWidth + int(leftMargin) + int(rightMargin), 3)
    mask = numpy.zeros(transImg.shape[0], transImg.shape[1])
    for i in range(cropHeight):
        for j in range(cropWidth):
            x = int(math.floor(transImgMap[i][j][0])) + leftMargin
            y = int(math.floor(transImgMap[i][j][1])) + topMargin
            if (x < 0 or y < 0):
                print "i = {}, j = {}, x = {}, y = {}".format(i, j, x, y)
                break
            transImg[y, x] = fileImg[i + yMin, j + xMin]
            mask[y, x] = 1
    
    fullSize = 128
    fullSizeImg = numpy.zeros((fullSize, fullSize, 3))
    left = (fullSize - transImg.shape[1])/2
    top = (fullSize - transImg.shape[0])/2

    fullSizeImg[top : transImg.shape[0] + top, left : transImg.shape[1] + left, :] = transImg
    img = Image.fromarray(fullSizeImg.astype('uint8'))
    img.save(file + "_front.jpg")

    fullMask = numpy.zeros((fullSize, fullSize))
    fullMask[top : transImg.shape[0] + top, left : transImg.shape[1] + left] = mask
    numpy.save(file + "_mask.npy", fullMask)



parser = argparse.ArgumentParser()

parser.add_argument('image')


args = parser.parse_args()

if (not os.path.exists(args.image)):
    print "{} not exists".format(args.image)
else:
    faceGen(args.image)