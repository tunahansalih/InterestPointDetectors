import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def read_image(filename):
    image = cv2.imread(filename)
    if image is None:
        print("Error reading")
    # show_image(image)
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
    k = 0.06
    window_size = 3
    threshold = 250
    x_gradient_kernel = np.zeros((1, 3))
    x_gradient_kernel[0, 0] = -1
    x_gradient_kernel[0, 2] = 1
    y_gradient_kernel = np.zeros((3, 1))
    y_gradient_kernel[0, 0] = -1
    y_gradient_kernel[2, 0] = 1
    dx = cv2.filter2D(image, -1, x_gradient_kernel)
    dy = cv2.filter2D(image, -1, y_gradient_kernel)
    # show_image(dx, gray=True)
    # show_image(dy, gray=True)

    Ixx = dx ** 2
    Ixy = dx * dy
    Iyy = dy ** 2
    corners = []

    sumIxx = cv2.filter2D(Ixx, -1, np.ones((window_size, window_size)))
    sumIxy = cv2.filter2D(Ixy, -1, np.ones((window_size, window_size)))
    sumIyy = cv2.filter2D(Iyy, -1, np.ones((window_size, window_size)))

    det_m = sumIxx * sumIyy - sumIxy * sumIxy
    trace_m = (sumIxx + sumIyy)
    r_values = det_m - k * (trace_m ** 2)

    for y in range(r_values.shape[0]):
        for x in range(r_values.shape[1]):
            if r_values[y, x] > threshold:
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

    return np.array(harris_points), np.array(sift_points), sift_keypoints, np.array(surf_points), surf_keypoints


def measureRepeatability(keyPoints1, keyPoints2, homography1to2=np.diag(np.ones(3)), image2size=None):
    num_of_points_with_neighbours = 0
    img2 = np.zeros(image2size)
    keyPoints2 = cv2.perspectiveTransform(keyPoints2.reshape(-1, 1, 2).astype(np.float32), homography1to2)
    keyPoints2 = keyPoints2.reshape(-1, 2).astype(np.int32)
    for kp2 in keyPoints2:
        if img2.shape[0] > kp2[1] >= 0 and img2.shape[1] > kp2[0] >= 0:
            img2[kp2[1], kp2[0]] = 1

    for kp1 in keyPoints1:
        if img2.shape[0] - 2 > kp1[1] and kp1[1] >= 1 and img2.shape[1] - 2 > kp1[0] and kp1[0] >= 1:
            if np.sum(img2[kp1[1] - 1:kp1[1] + 2, kp1[0] - 1:kp1[0] + 2]) > 0:
                num_of_points_with_neighbours += 1

    repeatability = num_of_points_with_neighbours / np.min((len(keyPoints1), len(keyPoints2)))
    return repeatability


graffiti_image_files = [f'Images/img{i}.png' for i in [1, 2, 3, 4, 5, 6]]
noisy_image_files = [f'Images/kuzey_noise_mean_{x}.jpeg' for x in [5, 10, 20, 50, 100, 200]]
jpeg_image_files = [f'Images/kuzey_quality_{quality}.jpeg' for quality in [0, 20, 40, 60, 20, 100]]
homography_matrix_files = [f'Images/H1to{x}p' for x in range(2, 7)]

filename = "Images/kuzey.jpg"
create_different_quality_images(filename)
create_different_images_with_gaussian(filename)

img1 = read_image(filename)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
harris_points1, sift_points1, sift_keypoints1, surf_points1, surf_keypoints1 = get_harris_sift_surf_points(img1_gray)

image_with_points = img1.copy()

for x, y in harris_points1:
    image_with_points[y, x] = [0, 0, 255]

image_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Image with Detected Corners Using Harris Algorithm")
plt.savefig('Images/harris_corners.png')

image_with_keypoints = cv2.drawKeypoints(img1, sift_keypoints1, None)
img_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Image with Detected Corners Using SIFT algorithm")
plt.savefig('Images/sift_corners.png')

image_with_keypoints = cv2.drawKeypoints(img1, surf_keypoints1, None)
img_rgb = cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Image with Detected Corners Using SURF algorithm")
plt.savefig('Images/surf_corners.png')

noisy_repeatability = {'harris': [], 'sift': [], 'surf': []}
for i in range(len(noisy_image_files)):
    img2 = read_image(noisy_image_files[i])
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    harris_points2, sift_points2, sift_keypoints2, surf_points2, surf_keypoints2 = get_harris_sift_surf_points(
        img2_gray)

    harris_repeatability = measureRepeatability(keyPoints1=harris_points1, keyPoints2=harris_points2,
                                                image2size=img2_gray.shape)
    sift_repeatability = measureRepeatability(keyPoints1=sift_points1, keyPoints2=sift_points2,
                                              image2size=img2_gray.shape)
    surf_repeatability = measureRepeatability(keyPoints1=surf_points1, keyPoints2=surf_points2,
                                              image2size=img2_gray.shape)
    noisy_repeatability['harris'].append(harris_repeatability)
    noisy_repeatability['sift'].append(sift_repeatability)
    noisy_repeatability['surf'].append(surf_repeatability)

jpeg_repeatability = {'harris': [], 'sift': [], 'surf': []}
for i in range(len(jpeg_image_files)):
    img2 = read_image(jpeg_image_files[i])
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    harris_points2, sift_points2, sift_keypoints2, surf_points2, surf_keypoints2 = get_harris_sift_surf_points(
        img2_gray)
    harris_repeatability = measureRepeatability(keyPoints1=harris_points1, keyPoints2=harris_points2,
                                                image2size=img2_gray.shape)
    sift_repeatability = measureRepeatability(keyPoints1=sift_points1, keyPoints2=sift_points2,
                                              image2size=img2_gray.shape)
    surf_repeatability = measureRepeatability(keyPoints1=surf_points1, keyPoints2=surf_points2,
                                              image2size=img2_gray.shape)
    jpeg_repeatability['harris'].append(harris_repeatability)
    jpeg_repeatability['sift'].append(sift_repeatability)
    jpeg_repeatability['surf'].append(surf_repeatability)

img1 = read_image(graffiti_image_files[0])
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
harris_points1, sift_points1, sift_keypoints1, surf_points1, surf_keypoints1 = get_harris_sift_surf_points(img1_gray)

graffiti_repeatability = {'harris': [], 'sift': [], 'surf': []}
for i in range(1, len(graffiti_image_files)):
    img2 = read_image(graffiti_image_files[i])
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    harris_points2, sift_points2, sift_keypoints2, surf_points2, surf_keypoints2 = get_harris_sift_surf_points(
        img2_gray)
    homography_matrix = np.loadtxt(homography_matrix_files[i - 1])
    harris_repeatability = measureRepeatability(keyPoints1=harris_points1, keyPoints2=harris_points2,
                                                image2size=img2_gray.shape, homography1to2=homography_matrix)
    sift_repeatability = measureRepeatability(keyPoints1=sift_points1, keyPoints2=sift_points2,
                                              image2size=img2_gray.shape, homography1to2=homography_matrix)
    surf_repeatability = measureRepeatability(keyPoints1=surf_points1, keyPoints2=surf_points2,
                                              image2size=img2_gray.shape, homography1to2=homography_matrix)
    graffiti_repeatability['harris'].append(harris_repeatability)
    graffiti_repeatability['sift'].append(sift_repeatability)
    graffiti_repeatability['surf'].append(surf_repeatability)

for k, v in jpeg_repeatability.items():
    jpeg_rep = jpeg_repeatability[k]
    plt.plot(['0-1', '0-2', '0-3', '0-4', '0-5', '0-6'], jpeg_rep, '-o')
    plt.xlabel('Compared Images')
    plt.ylabel('Repeatability')
    plt.ylim([0, 1])
    plt.title(f'{k} Corner Detection on Different JPEG Qualities')
    plt.savefig(f'Images/{k}_jpeg.svg')

for k, v in noisy_repeatability.items():
    jpeg_rep = noisy_repeatability[k]
    plt.plot(['0-5', '0-10', '0-20', '0-50', '0-100', '0-200'], jpeg_rep, '-o')
    plt.xlabel('Compared Images')
    plt.ylabel('Repeatability')
    plt.ylim([0, 1])
    plt.title(f'{k} Corner Detection on Different Gaussian Noise')
    plt.savefig(f'Images/{k}_noisy.svg')

for k, v in graffiti_repeatability.items():
    jpeg_rep = graffiti_repeatability[k]
    plt.plot(['1-2', '1-3', '1-4', '1-5', '1-6'], jpeg_rep, '-o')
    plt.xlabel('Compared Images')
    plt.ylabel('Repeatability')
    plt.ylim([0, 1])
    plt.title(f'{k} Corner Detection on Homography Transformation')
    plt.savefig(f'Images/{k}_homography.svg')
