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


def draw_points(image, points, label=None):
    image_with_points = image.copy()

    for x, y in points:
        image_with_points[y, x] = [0, 0, 255]
    show_image(image_with_points, label=label)


def draw_keypoints(image, keypoints, label=None):
    img = cv2.drawKeypoints(image, keypoints, None)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
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
        return myHarrisCornerDetector(image), []
    elif algorithm == 'sift':
        print('Calculating SIFT Corners')
        sift_keypoints = cv2.xfeatures2d.SIFT_create().detect(image, None)
        corners = []
        for kp in sift_keypoints:
            corners.append([np.round(kp.pt[0]).astype(int), np.round(kp.pt[1]).astype(int)])
        return corners, sift_keypoints
    elif algorithm == 'surf':
        print('Calculating SURF Corners')
        surf_keypoints = cv2.xfeatures2d.SURF_create().detect(image, None)
        corners = []
        for kp in surf_keypoints:
            corners.append([int(kp.pt[0]), int(kp.pt[1])])
        return corners, surf_keypoints

    return []


def get_harris_sift_surf_points(image):
    harris_points, _ = get_corner_points(image, algorithm='harris')
    sift_points, sift_keypoints = get_corner_points(image, algorithm='sift')
    surf_points, surf_keypoints = get_corner_points(image, algorithm='surf')

    return harris_points, sift_points, sift_keypoints, surf_points, surf_keypoints


def measureRepeatability(keyPoints1, keyPoints2, homography1to2=np.fill_diagonal(np.zeros((3, 3)), 1), image2size=None):
    num_of_points_with_neighbours = 0
    img2 = np.zeros(image2size)
    for kp2 in keyPoints2:
        img2[kp2[1], kp2[0]] = 1

    for kp1 in keyPoints1:
        if np.sum(img2[kp1[1] - 1:kp1[1] + 2, kp1[0] - 1:kp1[0] + 2]) > 0:
            num_of_points_with_neighbours += 1

    repeatability = num_of_points_with_neighbours / np.min((len(keyPoints1), len(keyPoints2)))
    return repeatability


graffiti_image_files = [f'Images/img{i}.png' for i in [1, 2, 3, 4, 5, 6]]
noisy_image_files = [f'Images/kuzey_noise_mean_{x}.jpeg' for x in [5, 10, 20, 50, 100, 200]]
jpeg_image_files = [f'Images/kuzey_quality_{quality}.jpeg' for quality in np.linspace(0, 100, num=6, endpoint=True)]
homography_matrix_files = [f'Images/H1to{x}p' for x in range(2, 7)]
#
# filename = "Images/kuzey.jpg"
# create_different_quality_images(filename)
# create_different_images_with_gaussian(filename)
#
# image_bgr = read_image(filename)
# image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
#
# harris_points, sift_points, sift_keypoints, surf_points, surf_keypoints = get_harris_sift_surf_points(image_gray)
# draw_points(image_bgr, harris_points, label='harris')
# draw_keypoints(image_bgr, sift_keypoints, label='sift')
# draw_keypoints(image_bgr, surf_keypoints, label='surf')


img1 = read_image(noisy_image_files[0])
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
harris_points1, sift_points1, sift_keypoints1, surf_points1, surf_keypoints1 = get_harris_sift_surf_points(img1_gray)
for i in range(1, len(noisy_image_files)):
    img2 = read_image(noisy_image_files[i])
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    harris_points2, sift_points2, sift_keypoints2, surf_points2, surf_keypoints2 = get_harris_sift_surf_points(
        img2_gray)
    print(i, measureRepeatability(keyPoints1=harris_points1, keyPoints2=harris_points2, image2size=img2_gray.shape))
