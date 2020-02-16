import cv2
import numpy as np


# Object to store keypoint, descriptor and angle of a feature
class Feature:
    def __init__(self, point, descriptor, angle):
        self.keypoint = cv2.KeyPoint(point[1], point[0], 1)
        self.descriptor = descriptor
        self.angle = angle


# methods that create
def normalize(descriptor):
    sum = np.sqrt(np.sum(np.square(descriptor)))
    for x in range(len(descriptor)):
        descriptor[x] = descriptor[x] / sum
    return descriptor

def createfeatures(keypoints, height, width, img):
    print("Creating Features...")
    features = []
    for x, y in keypoints:
        # To evaluate if there is enough space to make a 16*16 around keypoint
        if x - 8 >= 0 and y - 8 >= 0 and x + 8 < height and y + 8 < width:
            feature = img[x - 8:x + 8, y - 8:y + 8]
            dx, dy = np.gradient(feature)
            dx = cv2.GaussianBlur(dx, (5, 5), 0, 0)
            dy = cv2.GaussianBlur(dy, (5, 5), 0, 0)
            angle = findpeak(hist(dx, dy, 36, 0), 36)  # finds dominant orientation, (can create multiple features
            descript = featuredescript(dx, dy, angle)  # finds descriptor for feature
            feature = Feature([x, y], descript, angle)  # creates feature object( keypoint + descriptor + angle)
            features.append(feature)
    print("Total of features detected in image: ", len(features))
    return features


# Calculates a descriptor for the feature using SIFT algorithm
def featuredescript(dx, dy, angle):
    descriptor = []
    gridsx = []
    gridsy = []

    # divide 16*16 into 16 4*4 grids
    for x in range(4):
        for y in range(4):
            gridsx.append(dx[x * 4:(x * 4 + 4), y * 4:(y * 4 + 4)])
            gridsy.append(dy[x * 4:(x * 4 + 4), y * 4:(y * 4 + 4)])

    # Calculate an 8 bin histogram for each grid and extend into 1 128d array
    for i in range(len(gridsx)):
        descriptor.extend((hist(gridsx[i], gridsy[i], 8, angle)))  # creating an 128 list/array

    # threshold normalize descriptor
    # divide each in descriptor by the squared root sum of the descriptor
    descriptor = normalize(descriptor)

    # This is for illumination invarience
    for i in range(len(descriptor)):
        if descriptor[i] > 0.2:
            descriptor[i] = 0.2

    # Need to normalize again after performing illumination invarience
    descriptor = normalize(descriptor)

    return descriptor


# finds the index with the max value of the histogram (did I want to try and find multiple with angle)
def findpeak(histogram, size):
    maximum = np.argmax(histogram)  # finds max index
    return maximum * (360 / size)  # need to multiply to find angle size


# Used to find dominant orientation/ key descriptor
# Creates histogram given the images dx,dy, size= number of bins, dom = dominent orientation/ if previously calculated
def hist(dx, dy, size, dom):

    histogram = np.zeros(size, dtype=np.float32)  # create histogram
    angles = np.arctan2(dy, dx)
    magnitude = np.sqrt(np.square(dy) + np.square(dx))
    angles = np.degrees(angles)

    # deals with negative angles and calculates new angle given dominent orientation
    for angle in angles:
        for x in range(len(angle)):
            angle[x] = angle[x] - dom
            if angle[x] < 0:
                angle[x] = 360 + angle[x]

    # place the magnitude at the proper index
    for i in range(len(angles)):
        for j in range(len(angles)):
            index = int(np.floor((angles[i, j]) / (360 / size)))
            if index > 8:
                index = 0
            histogram[index] += magnitude[i, j]  # number of degrees in each histogram 360/size

    return histogram
