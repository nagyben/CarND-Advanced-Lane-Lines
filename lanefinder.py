import calibration
import cv2
import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt
from line import Line

from moviepy.editor import VideoFileClip

SOBEL_MIN = 15
SOBEL_MAX = 150

S_MIN = 70
S_MAX = 225

H_MIN = 255
H_MAX = 255

L_MIN = 100
L_MAX = 255


def test_images():
    images = glob.glob('test_images/*.jpg')

    for fname in images:
        # Load image
        img = cv2.imread(fname)

        # Run pipeline
        processed_img = lanefinder_pipeline(img)
        output_image = processed_img

        # Save image
        output_name = 'output_images/{}'.format(os.path.basename(fname))
        cv2.imwrite(output_name, output_image)


def video():
    clip = VideoFileClip("challenge_video.mp4")
    output_clip = clip.fl_image(video_lanefinder_pipeline)
    output_name = 'challenge_video_output.mp4'
    output_clip.write_videofile(output_name, audio=False, threads=2)


def video_lanefinder_pipeline(img):
    return lanefinder_pipeline(img, video=True)
    # return test_pipeline(img)


def test_pipeline(img):
    """
    Pipeline to test thresholds
    :param img: The input image
    :return:
    """
    # Undistort
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    # Threshold on Sobel
    sobelx_bin = sobel_binary(undst, SOBEL_MIN, SOBEL_MAX)

    # Yellow and white mask
    yellow_mask = hsv_mask(undst, (0, 50), (100, 255), (100, 255))
    white_mask = hsv_mask(undst, (20, 255), (0, 80), (180, 255))

    # Combine binary thresholds
    comb_bin = np.zeros_like(sobelx_bin)

    # Combine masks and Sobel
    comb_bin[(yellow_mask == 1) | (white_mask == 1) | (sobelx_bin == 1)] = 1

    # Ground plane perspective warp
    warped_bin = cv2.warpPerspective(comb_bin, ground_plane_mtx, (img.shape[1], img.shape[0]))

    warped_bin = np.dstack((warped_bin, warped_bin, warped_bin)) * 255
    output_img = np.hstack((undst, warped_bin))

    return output_img


def lanefinder_pipeline(img, video=False):
    """
    The pipeline for laneline finding based on all the techniques used so far
    :param img: The image to find lanelines on
    :return: An image with laneline boundaries drawn on
    """
    # Undistort
    undst = cv2.undistort(img, mtx, dist, None, mtx)

    # Threshold on Sobel
    sobelx_bin = sobel_binary(undst, SOBEL_MIN, SOBEL_MAX)

    # Yellow and white mask
    yellow_mask = hsv_mask(undst, (0, 50), (100, 255), (100, 255))
    white_mask = hsv_mask(undst, (20, 255), (0, 80), (180, 255))

    # Combine binary thresholds
    comb_bin = np.zeros_like(sobelx_bin)

    # Combine masks and Sobel
    comb_bin[(yellow_mask == 1) | (white_mask == 1) | (sobelx_bin == 1)] = 1

    # Ground plane perspective warp
    warped_bin = cv2.warpPerspective(comb_bin, ground_plane_mtx, (img.shape[1], img.shape[0]))

    # Get some Lines
    if video:
        global leftLine, rightLine
        if leftLine is not None and leftLine.n_undetected < 5:
            leftLine = leftLine.search_from_prior(warped_bin, margin=50)
        else:
            # Get lane pixels using sliding windows
            leftx, lefty, rightx, righty, out_img = sliding_window(warped_bin)
            leftLine = Line(lefty, leftx)

        if rightLine is not None and rightLine.n_undetected < 5:
            rightLine = rightLine.search_from_prior(warped_bin, margin=50)
        else:
            # Get lane pixels using sliding windows
            leftx, lefty, rightx, righty, out_img = sliding_window(warped_bin)
            rightLine = Line(righty, rightx)
    else:
        # Get lane pixels using sliding windows
        leftx, lefty, rightx, righty, out_img = sliding_window(warped_bin)
        leftLine = Line(lefty, leftx)
        rightLine = Line(righty, rightx)

    ## Visualization ##
    # Create lane overlay image
    line_img = np.zeros_like(undst)

    # draw green shaded lane overlay
    lane_fill_outline = np.concatenate((leftLine.get_polyline(warped_bin.shape[0]),
                                        rightLine.get_polyline(warped_bin.shape[0])[:, ::-1, :]),
                                       axis=1)
    cv2.fillPoly(line_img, lane_fill_outline, color=(0, 255, 0))

    # draw left lane line
    cv2.polylines(line_img, leftLine.get_polyline(warped_bin.shape[0]), color=(255, 0, 0), isClosed=False,
                  thickness=8)
    # draw right lane line
    cv2.polylines(line_img, rightLine.get_polyline(warped_bin.shape[0]), color=(255, 0, 0), isClosed=False,
                  thickness=8)

    line_img_pre = cv2.addWeighted(line_img, 1, np.dstack((warped_bin, warped_bin, warped_bin)) * 255, 0.6, 0)

    # Get inverse transform matrix for drawing on original image
    _, inverse_ground_plane_mtx = cv2.invert(ground_plane_mtx)
    line_img = cv2.warpPerspective(line_img, inverse_ground_plane_mtx, (line_img.shape[1], line_img.shape[0]))

    # Combine images
    comb_bin = np.dstack((comb_bin, comb_bin, comb_bin)) * 255
    out_img = cv2.addWeighted(undst, 1, line_img, 0.5, 0)
    out_img = np.hstack((out_img, line_img_pre))

    return out_img


def sliding_window(warped_bin):
    histogram = np.sum(warped_bin[warped_bin.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((warped_bin, warped_bin, warped_bin)) * 255
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 8
    margin = 100
    minpix = 50
    window_height = np.int(warped_bin.shape[0] // nwindows)
    nonzero = warped_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    for window in range(nwindows):
        win_y_low = warped_bin.shape[0] - (window + 1) * window_height
        win_y_high = warped_bin.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def get_ground_plane_transform():
    """
    Function to test and get the ground plane transform
    :return: transform matrix to use with cv2.warpPerspective
    """
    img = cv2.imread('test_images/straight_lines1.jpg')

    # manually determined using image plotter
    road_plane = np.array([[195, img.shape[0]],
                           [600, 446],
                           [680, 446],
                           [1124, img.shape[0]]], np.float32)

    dest_plane = np.array([[350, img.shape[0]],
                           [350, 0],
                           [900, 0],
                           [900, img.shape[0]]], np.float32)

    t_mtx = cv2.getPerspectiveTransform(road_plane, dest_plane)
    img_size = (img.shape[1], img.shape[0])
    perspective_transform_img = cv2.warpPerspective(img, t_mtx, img_size, flags=cv2.INTER_LINEAR)

    cv2.polylines(img, np.int32([road_plane]), True, (0, 0, 255), thickness=2)
    cv2.polylines(perspective_transform_img, np.int32([dest_plane]), True, (0, 0, 255), thickness=2)
    output_img = np.hstack((img, perspective_transform_img))

    cv2.imwrite('writeup_images/ground_plane_transform.jpg', output_img)
    return t_mtx


def sobel_binary(img, sobel_min=0, sobel_max=255):
    """
    Returns the thresholded Sobel binary
    :param img: The image to run the filter on
    :param sobel_min: minimum Sobel threshold
    :param sobel_max: maximum Sobel threshold
    :return: binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel filter (absolute and scaled)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobelx = np.abs(sobelx)
    sobelx = np.uint8(255 * sobelx / np.max(sobelx))
    sobelx_bin = np.zeros_like(sobelx)
    sobelx_bin[(sobelx >= sobel_min) & (sobelx <= sobel_max)] = 1
    return sobelx_bin


def h_binary(img, h_min=0, h_max=255):
    """
    Returns the thresholded hue-channel binary of an image
    :param img: The image to run the filter on
    :param h_min: The minimum hue threshold
    :param h_max: The maximum hue threshold
    :return: binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_ch = hls[:, :, 0]
    h_bin = np.zeros_like(h_ch)
    h_bin[(h_ch >= h_min) & (h_ch <= h_max)] = 1
    return h_bin


def s_binary(img, s_min=0, s_max=255):
    """
    Returns the thresholded saturation-channel binary of an image
    :param img: The image to run the filter on
    :param s_min: The minimum saturation threshold
    :param s_max: The maximum saturation threshold
    :return: binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_ch = hls[:, :, 2]
    s_bin = np.zeros_like(s_ch)
    s_bin[(s_ch >= s_min) & (s_ch <= s_max)] = 1
    return s_bin


def l_binary(img, l_min=0, l_max=255):
    """
    Returns the thresholded lightness-channel binary of an image
    :param img: The image to run the filter on
    :param l_min: The minimum lightness value
    :param l_max: The maximum lightness value
    :return: binary image
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_ch = hls[:, :, 1]
    l_bin = np.zeros_like(l_ch)
    l_bin[(l_ch >= l_min) & (l_ch <= l_max)] = 1
    return l_bin


def v_binary(img, v_min=0, v_max=255):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_ch = hsv[:, :, 2]
    v_bin = np.zeros_like(v_ch)
    v_bin[(v_ch >= v_min) & (v_ch <= v_max)] = 1
    return v_bin


def hsv_mask(img, hue_mask, sat_mask, val_mask):
    """
    Returns a binary image based on the mask thresholds
    :param img: The image to mask
    :param hue_mask: Tuple of (hue_min, hue_max)
    :param sat_mask: Tuple of (sat_min, sat_max)
    :param val_mask: Tuple of (val_min, val_max
    :return: Binary image mask
    """
    hue_mask = h_binary(img, hue_mask[0], hue_mask[1])
    sat_mask = s_binary(img, sat_mask[0], sat_mask[1])
    val_mask = v_binary(img, val_mask[0], val_mask[1])
    mask = np.zeros_like(hue_mask)
    mask[(hue_mask == 1) & (sat_mask == 1) & (val_mask == 1)] = 1
    return mask


# Get camera undistortion matrices
_, mtx, dist, _, _ = calibration.calibrateCamera()

# Get ground plane transformation matrix
ground_plane_mtx = get_ground_plane_transform()

# global laneline variables for video
leftLine = None
rightLine = None

if __name__ == "__main__":
    video()
