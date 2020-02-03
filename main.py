import cv2
import numpy as np


def computeHarris(img, window, threshold):
    # the window size is how much pixels we take into account
    height, width = img.shape

    #   take derivatives of image
    dx, dy = np.gradient(img)

    # Get squares of derivatives
    dx2 = dx*dx
    dy2 = dy*dy
    dxy = dx*dy
    # smooth each with gaussian
    x3 = cv2.GaussianBlur(dx2, (5,5),0,0)  # We also should specify the standard deviation in the X and Y directions,??
    y3 = cv2.GaussianBlur(dy2, (5,5),0,0)   # since gaussian has both x and y directions.. will it make a difference
    xy = cv2.GaussianBlur(dxy, (5,5),0,0)

    #   for each element in the matrix calculate it's cropped derivatives using window size
    win = int(window/2)  # if window of 5
    corner = img.copy()
    a = []
    for x in range(win, height - win):
        for y in range(win, width - win):
            newdx = x3[x-win:x+win, y-win:y+win]
            newdy = y3[x-win:x+win, y-win:y+win]
            newxy = xy[x-win:x+win, y-win:y+win]
            newdx = newdx.sum()
            newdy = newdy.sum()
            newxy = newxy.sum()
            # calculate det and trace
            det = newdx*newdy - newxy*newxy
            trace = newdx + newdy
            c = det/trace
            corner[x, y] = c
            print(c)
            if(c > threshold):
                point = cv2.KeyPoint(x,y,1,-1)
                a.append(point)

    #take absolute value before converting back to uint8
    img = img.astype(np.uint8)

    outImage = cv2.drawKeypoints(img, a, img, color=(0,255,0), flags=0)
    cv2.imshow("keypoints", outImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #   Part 1: Feature Detection
    img = cv2.imread('image_sets/yosemite/Yosemite1.jpg', 0).astype(np.float32)
    win = 5 / 2
    print(np.round(win))
    computeHarris(img, 5, 500)
    #   identify points of interest in the image using the harris corner detection
    #   for each point in the image consider a window of pixels around that point
    #   compute the harris marix H for that point (summation is over all pixels (u,v) window
    #   weights chosed to be circularly symmetric (3*3 or 5*5 gaussian mask)
    #   interest points compute the corner strength function c(H)


#   Part 2: Feature Description
#   Part 3: Feature Matching

