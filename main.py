import cv2
from Feature import Feature
import Feature
import numpy as np


#   Function to compute the Harris Corner algorithm to find identify keypoints in image
#   Returns a list of keypoints
def computeHarris(img, window, threshold):
    print("Finding Keypoints...")
    height, width = img.shape

    #   take derivatives of image
    dx, dy = np.gradient(img)  # test using sobel instead

    # Get squares of derivatives
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # Smooth each with gaussian
    dx2 = cv2.GaussianBlur(dx2, (5, 5), 0, 0)
    dy2 = cv2.GaussianBlur(dy2, (5, 5), 0, 0)
    dxy = cv2.GaussianBlur(dxy, (5, 5), 0, 0)

    #   take a window over the derivatives, sum and then calculate the corner strength function
    win = int(window / 2)
    rangec = np.zeros((height, width), dtype=int)
    keypoints = []
    for x in range(win, height - win):
        for y in range(win, width - win):
            newdx = dx2[x - win:x + win + 1, y - win:y + win + 1]
            newdy = dy2[x - win:x + win + 1, y - win:y + win + 1]
            newxy = dxy[x - win:x + win + 1, y - win:y + win + 1]
            newdx = newdx.sum()
            newdy = newdy.sum()
            newxy = newxy.sum()

            # Calculate corner strength function(C)
            # Calculate det and trace
            det = newdx * newdy - newxy * newxy
            trace = newdx + newdy
            c = det / (trace + 0.001)  # add very small number to denominator so we never divide by 0
            rangec[x, y] = c

    # Check if C is the maximum in a 3*3 neighbourhood
    for x in range(1, height):
        for y in range(1, width):
            neighbour = rangec[x - 1:x + 2, y - 1:y + 2]  # 3 by 3 neighbour hood max
            if rangec[x, y] == np.max(neighbour) and rangec[x, y] > threshold:
                keypoints.append((x, y))  # storing values of keypoints that are useful

    # return keypoints that are detected as corners
    print("Total of keypoints found: ", len(keypoints))
    return keypoints


# Calculating distance between two descriptors
def difference(feat1, feat2):
    ssd = 0
    for i in range(len(feat1)):
        ssd += np.square(feat1[i] - feat2[i])
    return ssd


# ratio test best match divided by second best match
def ratio(ssd1, ssd2):
    return ssd1 / ssd2


def matchfeatures(descript1, descript2):
    print("Matching Features...")
    low = np.inf
    low2 = np.inf
    keys1 = []
    keys2 = []
    indexmatch = 0

    for each in range(len(descript1)):
        for d in range(len(descript2)):
            dif = difference(descript1[each].descriptor, descript2[d].descriptor)
            if dif < low or dif < low2:
                if dif < low:
                    low = dif
                    indexmatch = d
                else:
                    low2 = dif
        test = ratio(low, low2)
        if test < 0.50:  # needs to be less than 0.80 or else error matches
            keys1.append(descript1[each].keypoint)
            keys2.append(descript2[indexmatch].keypoint)
        low = np.inf
        low2 = np.inf
    print("Total Number of Matches: ", len(keys1))
    return keys1, keys2


    # Display keypoints on image
def display(img1, img2, keys1, keys2, title):
    img1 = cv2.drawKeypoints(img1, keys1, None, color=(0, 255, 0), flags=0)
    img2 = cv2.drawKeypoints(img2, keys2, None, color=(0, 255, 0), flags=0)

    com = np.concatenate((img1.astype(np.uint8), img2.astype(np.uint8)), axis=1)
    cv2.imshow(title, com)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def displaymatches(descript1, descript2, keys1, keys2, img1, img2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(np.mat(descript1), np.mat(descript2))
    matches = sorted(matches, key=lambda x: x.distance)
    IMG = cv2.drawMatches(img1, keys1, img2, keys2, matches[:10], None, flags=2)

    cv2.imshow("Display matches", IMG)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #  Part 1: Feature Detection
    # Read in images
    img = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 0).astype(np.float32)
    img2 = cv2.imread('image_sets/yosemite/Yosemite2.jpg', 0).astype(np.float32)
    img3 = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 1).astype(np.uint8)
    img4 = cv2.imread('image_sets/yosemite/Yosemite2.jpg', 1).astype(np.uint8)

    # img = cv2.imread('image_sets/graf/img1.jpg', 0).astype(np.float32)
    # img2 = cv2.imread('image_sets/graf/img4.jpg', 0).astype(np.float32)
    # img3 = cv2.imread('image_sets/graf/img1.jpg', 1).astype(np.uint8)
    # img4 = cv2.imread('image_sets/graf/img4.jpg', 1).astype(np.uint8)

    height1, width1 = img.shape
    height2, width2 = img2.shape

    # Find keypoints in both images using harris corner detection
    keypoints1 = computeHarris(img, 5, 5000)
    keypoints2 = computeHarris(img2, 5, 5000)

    keypoints3 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints1]
    keypoints4 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints2]

    # Display keypoints found in images
    display(img3, img4, keypoints3, keypoints4, "Keypoints Detected")

    # Create features using SIFT around Keypoints; returns array of descriptors
    descriptors1 = Feature.createfeatures(keypoints1, height1, width1, img)
    descriptors2 = Feature.createfeatures(keypoints2, height2, width2, img2)

    # Compares descriptors from both images to find matches; Returns keys with best matches
    keys1, keys2 = matchfeatures(descriptors1, descriptors2)

    # Display matches
    display(img3, img4, keys1, keys2, "Matches Found")
