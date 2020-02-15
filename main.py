import cv2
import numpy as np


def computeHarris(img, window, threshold):
    # the window size is how much pixels we take into account
    height, width = img.shape

    #   take derivatives of image
    dx, dy = np.gradient(img)  # use sobel instead ?

    # Get squares of derivatives
    dx2 = dx*dx
    dy2 = dy*dy
    dxy = dx*dy

    # smooth each with gaussian
    x3 = cv2.GaussianBlur(dx2, (5, 5), 0, 0)
    y3 = cv2.GaussianBlur(dy2, (5, 5), 0, 0)
    xy = cv2.GaussianBlur(dxy, (5, 5), 0, 0)

    #   for each element in the matrix calculate it's cropped derivatives using window size
    win = int(window/2)  # if window of 5
    rangec = np.zeros((height, width), dtype=int)
    keypoints = []
    for x in range(win, height - win):
        for y in range(win, width - win):
            newdx = x3[x-win:x+win+1, y-win:y+win+1]
            newdy = y3[x-win:x+win+1, y-win:y+win+1]
            newxy = xy[x-win:x+win+1, y-win:y+win+1]
            newdx = newdx.sum()
            newdy = newdy.sum()
            newxy = newxy.sum()

            #corner strength function
            # calculate det and trace
            det = newdx*newdy - newxy*newxy
            trace = newdx + newdy
            c = det/(trace + 0.001)  # add very small number to denominator
            rangec[x, y] = c

    # want C to be the max in a 3 by 3 neighbourhood too 
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            neighbour = rangec[x - 1:x + 1 + 1, y - 1:y + 1 + 1]  # 3 by 3 neighbour hood max
            if rangec[x, y] == np.max(neighbour) and rangec[x, y] > threshold:
                keypoints.append((x, y))  # storing values of keypoints that are useful


    # return keypoints that are detected as corners
    return keypoints

# works to find dominant orientation/ key descriptor
def hist(dx, dy, size, dom):
    histogram = np.zeros(size, dtype=np.float32)  # create histogram
    angles = np.arctan2(dy, dx)
    magnitude = np.sqrt(np.square(dy) + np.square(dx))
    angles = np.degrees(angles)

    for angle in angles:
        for x in range(len(angle)):
            angle[x] = angle[x] - dom
            if angle[x] < 0:
                angle[x] = 360 + angle[x]

    for i in range(len(angles)):
        for j in range(len(angles)):
            index = int(np.floor((angles[i, j] + 0.001)/(360/size)))
            if index == 8:
                index = 0
            histogram[index] += magnitude[i, j]  #number of degrees in each histogram 360/size
    print(histogram)
    return histogram

def findpeak(histogram, size):
    maximum = np.argmax(histogram) # finds max index
    return maximum * (360/size)   # need to multiply to find angle size


def featuredescript(dx, dy, feature, angle, ):
    # since we identified points of interest previously we need to come up with a descriptor for the feature centered at
    # each interest point
    # Start with a 16x16 window
    # divide the 16 x 16 window into a 4x4 grid of cells
    descriptor = []
    gridsx = []
    gridsy = []
    for x in range(4):
        for y in range(4):
            gridsx.append(dx[x*4:(x*4+4), y*4:(y*4+4)])
            gridsy.append(dy[x*4:(x*4+4), y*4:(y*4+4)])
            print("should be 4*4 cube")
            print(gridsx[x].shape)

    for i in range(len(gridsx)):
        descriptor.extend((hist(gridsx[i], gridsy[i], 8, angle)))  # creating an 128 list/array

    # threshold normalize descriptor
    sum = np.sqrt(np.sum(np.square(descriptor)))
    for x in range(len(descriptor)):
        descriptor[x] = descriptor[x]/sum

    for i in range(len(descriptor)):
        if descriptor[i] > 0.2:
            descriptor[i] = 0.2

    sum = np.sqrt(np.sum(np.square(descriptor))) # normalize again
    for x in range(len(descriptor)):
        descriptor[x] = descriptor[x] / sum

    print(descriptor)
    return descriptor


def difference(feat1, feat2):
    ssd = 0
    for i in range(len(feat1)):
        ssd += np.square(feat1[i] - feat2[i])
    return ssd


# ratio test best match divided by second best match
def ratio(ssd1, ssd2):
    return ssd1/ssd2


def creatingfeature(keypoints, height, width, img):
    descriptors = []
    features = []
    for x, y in keypoints:
        if x -8 >= 0 and y-8 >= 0 and x+8 < height and y+8 < width: # making sure there is enough room around to make feature
            print("Keypoint :", y, x)
            feature = img[x-8:x+8, y-8:y+8]
            print("feature shape: should be 16*16")
            print(feature.shape)
            dx, dy = np.gradient(feature)
            dx = cv2.GaussianBlur(dx, (5, 5), 0, 0)
            dy = cv2.GaussianBlur(dy, (5, 5), 0, 0)
            angle = findpeak(hist(dx, dy, 36, 0), 36)    # finds dominant orientation
            features.append(feature)
            descriptors.append(featuredescript(dx, dy, feature, angle,).append([x, y]))
    return descriptors


if __name__ == '__main__':
    #   Part 1: Feature Detection
    img = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 0).astype(np.float32)
    img2 = cv2.imread('image_sets/yosemite/Yosemite2.jpg', 0).astype(np.float32)
    img3 = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 1).astype(np.uint8)
    img4 = cv2.imread('image_sets/yosemite/Yosemite2.jpg', 1).astype(np.uint8)
    win = 5 / 2
    keypoints1 = computeHarris(img, 5, 5000)
    keypoints2 = computeHarris(img2, 5, 5000)
    height1, width1 = img.shape
    height2, width2 = img2.shape
    descriptors1 = creatingfeature(keypoints1, height1, width1, img)
    descriptors2 = creatingfeature(keypoints2, height2, width2, img2)
    keypoints1 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints1]
    keypoints2 = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints2]
    img3 = cv2.drawKeypoints(img3, keypoints1, None, color=(0, 255, 0), flags=0)
    img4 = cv2.drawKeypoints(img4, keypoints2, None, color=(0, 255, 0), flags=0)

    com = np.concatenate((img3.astype(np.uint8), img4.astype(np.uint8)), axis=1)
    cv2.imshow("matches", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("matches", img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    breakpoint()

    low = np.inf
    low2 = np.inf
    match = []
    match2 = []
    indexmatch = 0
    for each in range(len(descriptors1)):
        for d in range(len(descriptors2)):
            dif = difference(descriptors1[each], descriptors2[d])
            if dif < low or dif < low2:
                if dif < low:
                    low = dif
                    indexmatch = d
                else:
                    low2 = dif
        test = ratio(low, low2)
        if test < 0.05:
            print(test)
            match.append((each, indexmatch))
            match2.append((descriptors1[each], descriptors2[indexmatch]))
        low = np.inf
        low2 = np.inf

    keys1 = []
    keys2 = []
    for x, y in match:
        keys1.append([keypoints1[x][0], keypoints1[x][1]])
        keys2.append([keypoints2[y][0], keypoints2[y][1]])
        # img3[keypoints1[x][0], keypoints1[x][1]] = [0, 0, 255]  # i know this works
        # img4[keypoints2[y][0], keypoints2[y][1]] = [0, 0, 255]  # i know my keypoints/ corner detection is working

    keys1 = [cv2.KeyPoint(x[1], x[0], 1) for x in keys1]
    keys2 = [cv2.KeyPoint(x[1], x[0], 1) for x in keys2]
    img3 = cv2.drawKeypoints(img3, keys1, None, color=(0, 255, 0), flags=0)
    img4 = cv2.drawKeypoints(img4, keys2, None, color=(0, 255, 0), flags=0)

    com = np.concatenate((img3.astype(np.uint8), img4.astype(np.uint8)), axis=1)
    cv2.imshow("matches", com)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    com = np.concatenate((img3.astype(np.uint8), img4.astype(np.uint8)), axis=1)
    cv2.imshow("matches", com)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

