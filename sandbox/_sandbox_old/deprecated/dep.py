
"""
deprecated Methods:
"""

# class Detector:
#     """
#     Detector for Objects or Markers in a specified Region, based on the RGB image from a kinect
#     """
#     #TODO: implement area of interest!
#     def __init__(self):
#
#         self.shapes=None
#         self.circles=None
#         self.circle_coords=None
#         self.shape_coords=None
#
#         #default parameters for the detection function:
#         self.thresh_value=80
#         self.min_area=30
#
#
#     def where_shapes(self,image, thresh_value=None, min_area=None):
#         """Get the coordinates for all detected shapes.
#
#                 Args:
#                     image (image file): Image input.
#                     min_area (int, float): Minimal area for a shape to be detected.
#                 Returns:
#                     x- and y- coordinates for all detected shapes as a 2D array.
#
#             """
#         if thresh_value is None:
#             thresh_value = self.thresh_value
#         if min_area is None:
#             min_area=self.min_area
#
#         bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
#         gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]
#         edge_detected_image = cv2.Canny(thresh, 75, 200)
#
#         _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         contour_list = []
#         contour_coords = []
#         for contour in contours:
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
#             else:
#                 cX = 0
#                 cY = 0
#
#             approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
#             area = cv2.contourArea(contour)
#             if ((len(approx) > 8) & (len(approx) < 23) & (area > min_area)):
#                 contour_list.append(contour)
#                 contour_coords.append([cX, cY])
#         self.shapes=numpy.array(contour_coords)
#
#     def where_circles(self, image, thresh_value=None):
#         """Get the coordinates for all detected circles.
#
#                     Args:
#                         image (image file): Image input.
#                         thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
#                     Returns:
#                         x- and y- coordinates for all detected circles as a 2D array.
#
#                 """
#         if thresh_value is None:
#             thresh_value = self.thresh_value
#         #output = image.copy()
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)[1]
#         # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2 100)
#         circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 2, numpy.array([]), 200, 8, 4, 8)
#
#         if circles != [] and circles is not None:
#             # convert the (x, y) coordinates and radius of the circles to integers
#             circles = numpy.round(circles[0, :]).astype("int")
#             # print(circles)
#             circle_coords = numpy.array(circles)[:, :2]
#             dist = scipy.spatial.distance.cdist(circle_coords, circle_coords, 'euclidean')
#             #minima = np.min(dist, axis=1)
#             dist_bool = (dist > 0) & (dist < 5)
#             pos = numpy.where(dist_bool == True)[0]
#             grouped = circle_coords[pos]
#             mean_grouped = (numpy.sum(grouped, axis=0) / 2).astype(int)
#             circle_coords = numpy.delete(circle_coords, list(pos), axis=0)
#             circle_coords = numpy.vstack((circle_coords, mean_grouped))
#
#             self.circles = circle_coords.tolist()
#
#
#
#     def filter_circles(self, shape_coords, circle_coords):
#         dist = scipy.spatial.distance.cdist(shape_coords, circle_coords, 'euclidean')
#         minima = numpy.min(dist, axis=1)
#         non_circle_pos = numpy.where(minima > 10)
#         return non_circle_pos
#
#     def where_non_circles(self, image, thresh_value=None, min_area=None):
#         if thresh_value is None:
#             thresh_value = self.thresh_value
#         if min_area is None:
#             min_area=self.min_area
#         shape_coords = self.where_shapes(image, thresh_value, min_area)
#         circle_coords = self.where_circles(image, thresh_value)
#         if len(circle_coords)>0:
#             non_circles = self.filter_circles(shape_coords, circle_coords)
#             return shape_coords[non_circles].tolist()  #ToDo: what is this output?
#         else:
#             return shape_coords.tolist()
#
#     def get_shape_coords(self, image, thresh_value=None, min_area=None):
#         """Get the coordinates for all shapes, classified as circles and non-circles.
#
#                         Args:
#                             image (image file): Image input.
#                             thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
#                             min_area (int, float): Minimal area for a non-circle shape to be detected.
#                         Returns:
#                             x- and y- coordinates for all detected shapes as 2D arrays.
#                             [0]: non-circle shapes
#                             [1]: circle shapes
#
#                     """
#         if thresh_value is None:
#             thresh_value = self.thresh_value
#         if min_area is None:
#             min_area=self.min_area
#         non_circles = self.where_non_circles(image, thresh_value, min_area)
#         circles = self.where_circles(image, thresh_value)
#
#         return non_circles, circles
#
#
#     def plot_all_shapes(self, image, thresh_value=None, min_area=None):
#         """Plot detected shapes onto image.
#
#                             Args:
#                                 image (image file): Image input.
#                                 thresh_value (int, optional, default = 80): Define the lower threshold value for shape recognition.
#                                 min_area (int, float): Minimal area for a non-circle shape to be detected.
#
#                         """
#         if thresh_value is None:
#             thresh_value = self.thresh_value
#         if min_area is None:
#             min_area=self.min_area
#
#         output = image.copy()
#         non_circles, circles = self.get_shape_coords(image, thresh_value, min_area)
#         for (x, y) in circles:
#             cv2.circle(output, (x, y), 5, (0, 255, 0), 3)
#             # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#         for (x, y) in non_circles:
#             cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#         out_image = numpy.hstack([image, output])
#         plt.imshow(out_image)
#
#     def non_circles_fillmask(self, image, th1=60, th2=80):   #TODO: what is this function?
#         bilateral_filtered_image = cv2.bilateralFilter(image, 5, 175, 175)
#         gray = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#         thresh = cv2.threshold(blurred, th1, 1, cv2.THRESH_BINARY)[1]
#         circle_coords = self.where_circles(image, th2)
#         for (x, y) in circle_coords:           cv2.circle(thresh, (x, y), 20, 1, -1)
#         return numpy.invert(thresh.astype(bool))
#
#
# class Terrain:
#     """
#     simple module to visualize the topography in the sandbox with contours and a colormap.
#     """
#     def __init__(self,calibration=None, cmap='terrain', contours=True):
#         """
#         :type contours: boolean
#         :type cmap: matplotlib colormap object or keyword
#         :type calibration: Calibration object. By default the last created calibration is used.
#
#         """
#         if calibration is None:
#             try:
#                 self.calibration = Calibration._instances[-1]
#                 print("using last calibration instance created: ", calibration)
#             except:
#                 print("no calibration found")
#                 self.calibration = calibration
#
#         self.cmap = cmap
#         self.contours = contours
#         self.main_levels = numpy.arange(0, 2000, 50)
#         self.sub_levels = numpy.arange(0, 2000, 10)
#
#
#     def setup(self):
#         pass
#
#     def render_frame(self,depth):
#         depth_rotated = scipy.ndimage.rotate(depth, self.calibration.calibration_data.rot_angle, reshape=False)
#         depth_cropped = depth_rotated[self.calibration.calibration_data.y_lim[0]:self.calibration.calibration_data.y_lim[1],
#                         self.calibration.calibration_data.x_lim[0]:self.calibration.calibration_data.x_lim[1]]
#         depth_masked = numpy.ma.masked_outside(depth_cropped, self.calibration.calibration_data.z_range[0],
#                                                self.calibration.calibration_data.z_range[
#                                                    1])  # depth pixels outside of range are white, no data pixe;ls are black.
#
#         h = self.calibration.calibration_data.scale_factor * (
#                     self.calibration.calibration_data.y_lim[1] - self.calibration.calibration_data.y_lim[0]) / 100.0
#         w = self.calibration.calibration_data.scale_factor * (
#                 self.calibration.calibration_data.x_lim[1] - self.calibration.calibration_data.x_lim[0]) / 100.0
#
#         fig = plt.figure(figsize=(w, h), dpi=100, frameon=False)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         if self.contours is True:
#             x = range(numpy.shape(depth_cropped)[1])
#             y = range(numpy.shape(depth_cropped)[0])
#             z = depth_cropped
#             sub_contours = plt.contour(x, y, z, levels=self.sub_levels, linewidths=0.5, colors=[(0, 0, 0, 0.8)])
#             main_contours = plt.contour(x, y, z, levels=self.main_levels, linewidths=1.0, colors=[(0, 0, 0, 1.0)])
#             plt.clabel(main_contours, inline=0, fontsize=15, fmt='%3.0f')
#         ax.pcolormesh(depth_masked, vmin=self.calibration.calibration_data.z_range[0],
#                       vmax=self.calibration.calibration_data.z_range[1], cmap=self.cmap)
#         plt.savefig('current_frame.png', pad_inches=0)
#         plt.close(fig)
#

