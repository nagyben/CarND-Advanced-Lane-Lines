import cv2
import numpy as np
import glob
import os


def calibrateCamera():
    """
    Returns the camera matrix, distortion coefficients, rvecs and tvecs based on the calibration
    images in /camera_cal.
    Writes images to /camera_cal_output for validation and example purposes.
    :return: tuple of (ret, mtx, dist, rvecs, tvecs)
    """
    # Checkerboard dimensions
    nx = 9
    ny = 6

    # Prepare object points like [0,0,1], [1,0,0] etc.
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Initialize object and image points for calibration images
    objpoints = []
    imgpoints = []

    images = glob.glob('camera_cal/calibration*.jpg')
    for fname in images:
        # Read image
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            # Append to objectpoints and imgpoints
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw chessboard corners on image
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    # Assuming all images are the same size, it should be fine to call this on the last img
    img_size = (img.shape[1], img.shape[0])

    # Get the camera calibration matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Test undistortion on calibration images
    OUTPUT_DIR = 'camera_cal_output'
    for fname in images:
        # Load image
        img = cv2.imread(fname)

        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        # Stack original and undistorted
        out = np.hstack((img, dst))

        # Create output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Get output path
        outname = '{}/{}'.format(OUTPUT_DIR, os.path.basename(fname))

        # Save image
        cv2.imwrite(outname, out)

    return ret, mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    calibrateCamera()

