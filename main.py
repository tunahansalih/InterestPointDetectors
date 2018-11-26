import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def read_image(filename):
    image = cv2.imread(filename)
    show_image(image)
    return image


def show_image(image, gray=False, label=None):
    if gray:
        plt.imshow(image, cmap='gray')

    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    if label is not None:
        plt.title(label=label)
    plt.show()


def myHarrisCornerDetector(image):
    k = 0.04
    window_size = 5
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    show_image(dx, gray=True)
    show_image(dy, gray=True)

    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2
    image_height = image.shape[0]
    image_width = image.shape[1]
    corners = []
    r_values = np.zeros_like(image, dtype=np.float32)
    offset = window_size // 2

    for y in range(offset, image_height - offset):
        for x in range(offset, image_width - offset):
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            sumIxx = np.sum(windowIxx)
            sumIxy = np.sum(windowIxy)
            sumIyy = np.sum(windowIyy)

            M = np.array([np.array([sumIxx, sumIxy]),
                          np.array([sumIxy, sumIyy])])

            r = np.linalg.det(M) - k * (np.trace(M) ** 2)
            r_values[y, x] = r

    max_r_val = np.max(r_values)
    for y in range(r_values.shape[0]):
        for x in range(r_values.shape[1]):
            if r_values[y, x] > 0.01 * max_r_val:
                corners.append([x, y])
    return corners


def create_different_quality_images(filename):
    image = cv2.imread(filename)
    for quality in np.linspace(0, 100, num=6, endpoint=True):
        cv2.imwrite(f'Images/{os.path.splitext(os.path.basename(filename))[0]}_quality_{int(quality)}.jpeg', image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def create_different_images_with_gaussian(filename):
    image = cv2.imread(filename)
    for std in [5, 10, 20, 50, 100, 200]:
        noise = np.random.normal(loc=0.0, scale=std, size=image.shape)
        image = image + noise
        cv2.imwrite(f'Images/{os.path.splitext(os.path.basename(filename))[0]}_noise_mean_{std}.jpeg', image)


def get_corner_points(image, algorithm):
    if algorithm == 'harris':
        print('Calculating Harris Corners')
        return myHarrisCornerDetector(image)
    elif algorithm == 'sift':
        print('Calculating SIFT Corners')
        sift_keypoints = cv2.xfeatures2d.SIFT_create().detect(image, None)
        corners = []
        for kp in sift_keypoints:
            corners.append([np.round(kp.pt[0]).astype(int), np.round(kp.pt[1]).astype(int)])
        return corners
    elif algorithm == 'surf':
        print('Calculating SURF Corners')
        surf_keypoints = cv2.xfeatures2d.SURF_create().detect(image, None)
        corners = []
        for kp in surf_keypoints:
            corners.append([int(kp.pt[0]), int(kp.pt[1])])
        return corners

    return []


def get_harris_sift_surf_points(image):
    harris_points = get_corner_points(image, algorithm='harris')
    sift_points = get_corner_points(image, algorithm='sift')
    surf_points = get_corner_points(image, algorithm='surf')

    return harris_points, sift_points, surf_points


def draw_points(image, points, label=None):
    image_with_points = image.copy()

    for x, y in points:
        image_with_points[y, x] = [0, 0, 255]
    show_image(image_with_points, label=label)


def measureRepeatability(keyPoints1, keyPoints2, homography1to2, image2size):
    pass


graffiti_images = [f'Images/img{i}.png' for i in [1, 2, 3, 4, 5, 6]]
noisy_images = [f'Images/kuzey_noise_mean_{x}' for x in [5, 10, 20, 50, 100, 200]]
jpeg_images = [f'Images/kuzey_quality_{quality}' for quality in np.linspace(0, 100, num=6, endpoint=True)]
filename = "Images/kuzey.jpg"
create_different_quality_images(filename)
create_different_images_with_gaussian(filename)
image_bgr = read_image(filename)
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

harris_points, sift_points, surf_points = get_harris_sift_surf_points(image_gray)
draw_points(image_bgr, harris_points, label='harris')
draw_points(image_bgr, sift_points, label='sift')
draw_points(image_bgr, surf_points, label='surf')
