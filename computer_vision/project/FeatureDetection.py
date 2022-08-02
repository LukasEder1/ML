import cv2
from skimage.feature import hog

def readImg(path, show = False, grayscale = False):

    image = cv2.imread(path)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if show:
        cv2.imshow("image", image)
        cv2.waitKey(0)

    return image

def SIFT(original, n = 150):

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=n)
    kp = sift.detect(original, None)
    kp, des = sift.compute(original, kp)
    #out = cv2.drawKeypoints(original, kp, original )
    if len(kp) != n:  # for some reason sometimes more than n keypoints are detected
        kp = kp[:n]
        des = des[:n]
    return kp, des

def SURF(original, n = 8000, n_keypoints = 50):

    surf = cv2.xfeatures2d.SURF_create(n)
    kp, des = surf.detectAndCompute(original, None)
    kp_sort = sorted(kp, key=lambda d: d.response)
    des_sort = [x for _, x in sorted(zip(kp, des), key=lambda d: d[0].response)]  # mirror kp_sort sorting to des_sort
    kp_sort = kp_sort[::-1]
    des_sort = des_sort[::-1]
    if(len(kp_sort) > n_keypoints):
        kp_sort = kp_sort[:n_keypoints]
        des_sort = des_sort[:n_keypoints]

    return kp_sort, des_sort

def HOG(original, pixels_per_cell, cells_per_block):
    feature_matrix8x8 = hog(original, orientations=9, pixels_per_cell=pixels_per_cell, 
                    cells_per_block=cells_per_block, visualize=False)
    return feature_matrix8x8
    
