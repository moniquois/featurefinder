import numpy as np


def matchfeatures(descript1, descript2, threshold):
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
        if test < threshold:  # needs to be less than 0.80 or else error matches
            keys1.append(descript1[each].keypoint)
            keys2.append(descript2[indexmatch].keypoint)
        low = np.inf
        low2 = np.inf
    print("Total Number of Matches: ", len(keys1))
    return keys1, keys2

# Calculating distance between two descriptors
def difference(feat1, feat2):
    ssd = 0
    for i in range(len(feat1)):
        ssd += np.square(feat1[i] - feat2[i])
    return ssd


# ratio test best match divided by second best match
def ratio(ssd1, ssd2):
    return ssd1 / ssd2
