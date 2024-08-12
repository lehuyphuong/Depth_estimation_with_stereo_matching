import cv2
import numpy as np


def distance(x, y):
    return abs(x - y)


def compute_depth(kernel_half, max_value, left, right, scale, y, x, disparity_range):
    depth = 0
    disparity = 0
    cost_min = 65534
    
    for j in range(disparity_range):
        total = 0
        value = 0
        for v in range(-kernel_half, kernel_half + 1):
            for u in range(-kernel_half, kernel_half + 1):
                value = max_value
                if (x + u - j >= 0):
                    value = distance(
                        int(left[y + v, x + u]),
                        int(right[y + v, (x + u) - j])
                    )

                total += value

                if total < cost_min:
                    cost_min = total
                    disparity = j

    depth = disparity * scale

    return depth


def window_based_matching_l1(left_imng, right_img,
                             disparity_range, kernel_size,
                             save_result=True):

    # Read left , right images then convert to grayscale
    left = cv2.imread(left_imng, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):

            depth[y, x] = compute_depth(
                kernel_half=kernel_half,
                max_value=max_value,
                left=left,
                right=right,
                scale=scale,
                y=y,
                x=x,
                disparity_range=disparity_range
            )

    if save_result == True:
        print("saving ...")
        cv2.imwrite("window_based_l1.png", depth)
        cv2.imwrite("window_based_l1_color.png",
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        print("Done")

    return depth


def window_based_matching_l2(left_img, right_img,
                             disparity_range, kernel_size,
                             save_result=True):

    # Read left , right images then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    kernel_half = int((kernel_size - 1) / 2)
    scale = 3
    max_value = 255 ** 2
    height, width = left.shape[:2]
    depth = np.zeros((height, width), np.uint8)

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):

            depth[y, x] = compute_depth(
                kernel_half=kernel_half,
                max_value=max_value,
                left=left,
                right=right,
                scale=scale,
                y=y,
                x=x,
                disparity_range=disparity_range
            )

    if save_result == True:
        print("saving ...")
        cv2.imwrite("window_based_l2.png", depth)
        cv2.imwrite("window_based_l2_color.png",
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))
        print("Done")

    return depth


if __name__ == "__main__":
    left_img_path = 'Aloe/Aloe_left_1.png'
    right_img_path = 'Aloe/Aloe_right_1.png'
    disparity_range = 64
    kener_size = 3

    window_based_result_l1 = window_based_matching_l1(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size=kener_size,
        save_result=True)

    window_based_result_l2 = window_based_matching_l2(
        left_img_path,
        right_img_path,
        disparity_range,
        kernel_size=kener_size,
        save_result=True)
