import numpy as np
import cv2

class Line:
    """
    Class used to store line information and the search_from_prior function
    """
    def __init__(self, x, y):
        """
        Creates a line with a polyfit of order 2 based on the x and y arguments
        :param x: The x-coordinate of the pixels
        :param y: The y-coordinate of the pixels
        """
        self.x = x
        self.y = y
        self.fit = np.polyfit(y, x, 2)
        self.detected = True
        self.n_undetected = 0

    def get_fit_polyline(self, img_height):
        """
        Returns a tuple of (x, y) coordinates of the fitted polyline
        :param img_height: The height of the image
        :return: tuple of (fitx, ploty)
        """
        ploty = np.linspace(0, img_height - 1, img_height)
        fitx = self.fit[0] * ploty ** 2 + self.fit[1] * ploty + self.fit[2]

        return fitx, ploty

    def search_from_prior(self, warped_bin, margin):
        """
        Returns a new line using the prior search method on the given image. If a new line is not found, it will return
        the current line
        :param warped_bin: The binary image to perform the search on
        :param margin: The margin to use for the search
        :return: a Line object with new parameters
        """
        nonzero = warped_bin.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = ((nonzerox > (self.fit[0] * (nonzeroy ** 2) + self.fit[1] * nonzeroy +
                                  self.fit[2] - margin)) & (nonzerox < (self.fit[0] * (nonzeroy ** 2) +
                                                                        self.fit[1] * nonzeroy + self.fit[
                                                                            2] + margin)))

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # If less than 5 pixels were detected, don't bother trying to calculate a new line
        if np.count_nonzero(x) < 5:
            return self

        try:
            newLine = Line(x, y)

            # EWMA filter for lane line smoothing
            alpha = 0.8
            newLine.fit = alpha * newLine.fit + (1 - alpha) * self.fit

            # Sanity check - if line is very different from last frame,
            # it's probably a bad detection - return the previous one
            coeff_threshold = 0.005
            if np.abs(newLine.fit[0] - self.fit[0]) > coeff_threshold:
                self.detected = False
                self.n_undetected += 1
                return self

            return newLine

        except TypeError:
            # If any error occured, just return the same line as the previous frame
            self.detected = False
            self.n_undetected += 1
            return self

    def get_curvature_pixels(self, img_height):
        """
        Returns the curvature of the road, in pixels, at the vehicle's current position
        :param img_height: The height of the image
        :return:
        """
        return ((1 + (2 * self.fit[0] * img_height + self.fit[1]) ** 2) ** 1.5) / np.abs(2 * self.fit[0])

    def get_curvature_meters(self, img_height, xm_per_pix, ym_per_pix):
        """
        Returns the curvature of the road, in meters, at the vehicle's current position
        Based on the x and y meters per pixel
        :param img_height: The image height
        :param xm_per_pix: The meters per pixel in the X axis
        :param ym_per_pix: The meters per pixed in the Y axis
        :return: The road curvature in meters
        """
        fit_cr = np.polyfit(self.y * ym_per_pix, self.x * xm_per_pix, 2)
        R_curve = ((1 + (2 * fit_cr[0] * img_height * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.abs(2 * fit_cr[0])
        return R_curve

    def get_polyline(self, img_height):
        """
        Returns the polylines for drawing the lane boundaries using cv.polyLine
        :param img_height:
        :return:
        """
        fitx, ploty = self.get_fit_polyline(img_height)
        fitx = np.array([fitx], np.int32).T
        ploty = np.array([ploty], np.int32).T
        return np.int32([np.concatenate((fitx, ploty), axis=1)])

    def get_base(self, img_height):
        """
        Returns the x-coordinate of the base of the line
        :return: integer
        """
        return np.int32(self.fit[0] * img_height ** 2 + self.fit[1] * img_height + self.fit[2])
