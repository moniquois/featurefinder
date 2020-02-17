import cv2
import Feature
import numpy as np
import HarrisCorner
import FeatureMatch as Match


    # Display keypoints on images side by side
def display(img1, img2, keys1, keys2, title):
    img1 = cv2.drawKeypoints(img1, keys1, None, color=(0, 255, 0), flags=0)
    img2 = cv2.drawKeypoints(img2, keys2, None, color=(0, 255, 0), flags=0)

    com = np.concatenate((img1.astype(np.uint8), img2.astype(np.uint8)), axis=1)
    cv2.imshow(title, com)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display the matches on the two images side by side
def displaymatches(descript1, descript2, keys1, keys2, img1, img2):
    descript1 = np.asarray(descript1, dtype=np.uint8)
    descript2 = np.asarray(descript2, dtype=np.uint8)
    bf = cv2.BFMatcher()
    # des is a numpy array of shape
    # number of key points * 128
    #matches = bf.knnMatch(descript1, descript2, k=2)
    matches = bf.match(descript1, descript2)

    #matches = sorted(matches, key=lambda x: x.distance)
    IMG = cv2.drawMatches(img1, keys1, img2, keys2, matches, None, flags=2)

    cv2.imshow("Display matches", IMG)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #  Part 1: Feature Detection
    # Read in images

    # Yosemite
    # img = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 0).astype(np.float32)
    # img2 = cv2.imread('image_sets/yosemite/Yosemite2.jpg', 0).astype(np.float32)
    # img3 = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 1).astype(np.uint8)
    # img4 = cv2.imread('image_sets/yosemite/Yosemite2.jpg', 1).astype(np.uint8)

    # Graf
    # img = cv2.imread('image_sets/graf/img1.jpg', 0).astype(np.float32)
    # img2 = cv2.imread('image_sets/graf/img2.jpg', 0).astype(np.float32)
    # img3 = cv2.imread('image_sets/graf/img1.jpg', 1).astype(np.uint8)
    # img4 = cv2.imread('image_sets/graf/img2.jpg', 1).astype(np.uint8)

    # Panorama
    img = cv2.imread('image_sets/panorama/pano1_0011.jpg', 0).astype(np.float32)
    img2 = cv2.imread('image_sets/panorama/pano1_0010.jpg', 0).astype(np.float32)
    img3 = cv2.imread('image_sets/panorama/pano1_0011.jpg', 1).astype(np.uint8)
    img4 = cv2.imread('image_sets/panorama/pano1_0010.jpg', 1).astype(np.uint8)

    # Panorama 2

    # img = cv2.imread('image_sets/panorama/pano1_0008.jpg', 0).astype(np.float32)
    # img2 = cv2.imread('image_sets/panorama/pano1_0009.jpg', 0).astype(np.float32)
    # img3 = cv2.imread('image_sets/panorama/pano1_0008.jpg', 1).astype(np.uint8)
    # img4 = cv2.imread('image_sets/panorama/pano1_0009.jpg', 1).astype(np.uint8)

    height1, width1 = img.shape
    height2, width2 = img2.shape

    # Find keypoints in both images using harris corner detection
    keypoints1 = HarrisCorner.computeHarris(img, 5, 5000)  # Use 3000 for lower threshold
    keypoints2 = HarrisCorner.computeHarris(img2, 5, 5000)

    keypoints3 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints1]
    keypoints4 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints2]

    # Display keypoints found in images
    display(img3, img4, keypoints3, keypoints4, "Keypoints Detected")

    # Create features using SIFT around Keypoints; returns array of descriptors
    descriptors1 = Feature.createfeatures(keypoints1, height1, width1, img)
    descriptors2 = Feature.createfeatures(keypoints2, height2, width2, img2)

    # Compares descriptors from both images to find matches; Returns keys with best matches
    keys1, keys2 = Match.matchfeatures(descriptors1, descriptors2, 0.4) # use 0.5 for higher threshold / above 0.75 creates false positives

    # Display matches
    display(img3, img4, keys1, keys2, "Matches Found")

    descriptors1 = [x.descriptor for x in descriptors1]
    descriptors2 = [x.descriptor for x in descriptors2]
    #displaymatches(descriptors1, descriptors2, img3, img4, keys1, keys2)
