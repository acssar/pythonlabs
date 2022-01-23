import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


def ghost_finder(ghost_image, work_picture, result_picture):
    """
    Function to find known objects (ghosts) on a picture.

    Parameters
    ----------
    ghost_image: numpy.ndarray
        image with object to find
    work_picture: numpy.ndarray
        picture where already found objects are cut
        so we can detect another objects
    result_picture: numpy.ndarray
        picture where found objects are in frames

    Returns
    -------
    1/0: int
        if a good matching object is found
    rect_image/work_picture: numpy.ndarray
        picture with (new) rectangle from polylines
    homography/result_picture: numpy.ndarray
        picture with (new) frame from polylines

    """
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ghost_image, None)
    kp2, des2 = orb.detectAndCompute(work_picture, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.77 * n.distance:
            good.append(m)
    # for m in matches:
    #     print(m.distance)
    if len(good) > 7:
        query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        # matches_mask = mask.ravel().tolist()
        # Perspective transform
        h, w = ghost_image.shape
        # -1 to cut correctly (avoid big white polygon)
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(result_picture, [np.int32(dst)], True, 255, 3)
        # cut out the found ghost
        a = dst[0][0]
        b = dst[1][0]
        c = dst[2][0]
        d = dst[3][0]
        sup = [max(a[0], b[0], c[0], d[0]), max(a[1], b[1], c[1], d[1])]
        inf = [min(a[0], b[0], c[0], d[0]), min(a[1], b[1], c[1], d[1])]

        for i in range(int(inf[0]), int(sup[0])):
            if i % 2 == 0:
                z = sup[1]
            else:
                z = inf[1]
            x = [[[i, z]]]
            dst = np.append(dst, x, axis=0)
        rect_image = cv2.polylines(work_picture, [np.int32(dst)], True, 255, 3)
        return 1, rect_image, homography
    else:
        return 0, work_picture, result_picture


def main():
    res_pic = cv2.imread('lab7.png', cv2.IMREAD_GRAYSCALE)
    cut_pic = cv2.imread('lab7.png', cv2.IMREAD_GRAYSCALE)
    candy_image = cv2.imread('candy_ghost.png', 0)
    pumpkin_image = cv2.imread('pumpkin_ghost.png', 0)
    scary_image = cv2.imread('scary_ghost.png', 0)
    images = [scary_image, candy_image, pumpkin_image]
    for im in images:
        while True:
            is_found, cut_pic, res_pic = ghost_finder(im, cut_pic, res_pic)
            if is_found == 0:
                is_found, cut_pic, res_pic = ghost_finder(cv2.flip(im, 1), cut_pic, res_pic)
                if is_found == 0:
                    break
            plt.imshow(res_pic, 'gray'), plt.show()


if __name__ == '__main__':
    main()
