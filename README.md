# featurefinder

Run main.py to start the program. Once the images pop up press any key to continue(ex: Space bar).
First the images will show side by side all the keypoints found using the Harris Corner Detection
Second the images will show all the matches found between the two images

The program is broken into three parts
1. Keypoint Detection
In the class HarrisCorner, method computeHarris() which finds the keypoints in an
image using the Harris Corner Algorithm

2. Feature Description
In the class Feature, features are created from given keypoints using SIFT algorithm. createfeatures() method creates the features
by taking a 16*16 window around keypoint, calculates the dominant orientation and featuredescript() method calculates the
descriptor for the feature

3. Feature Matching
In the class FeatureMatch, method matchfeatures() is used to compare descriptors using the ratio test to
find feature matches between two images