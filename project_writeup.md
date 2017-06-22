## Project 4 Writeup by Steven Eisinger

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_chess.png "Undistorted"
[image2]: ./output_images/undistorted_road.png "Road Transformed"
[image3]: ./output_images/color_grad_thresh.jpg "Binary Example"
[image4]: ./output_images/perspective_transform.jpg "Warp Example"
[image5]: ./output_images/find_lane_lines.jpg "Fit Visual"
[image6]: ./output_images/draw_lane_info.jpg "Radius of Curvature and Distance from Center"
[image7]: ./output_images/full_pipeline.jpg "Output"
[video1]: ./draw_lanes.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second and third code cells of the IPython notebook "lane_lines.ipynb".

To calibrate the camera, I generate a numpy array with 3Dcoordinates corresponding to each corner in real space, eg. (1,2,0) for the 3rd corner on the 2nd row. Next I use the chessboard images in camera_cal converted to grayscale with `cv2.findChessboardCorners()` to return an array of 2D pixel coordinates corresponding to the chessboard cell corners in the image (9x6 corners in this project, so a total of 54 points). These arrays are stored in objpoints and imgpoints, respectively, which are used by the `cv2.calibrateCamera()` to provide parameters for the `cv2.undistort()` function, which generates the undistorted image. An example can be seen below.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once I computed the mtx and dist parameters in camera calibration, I can use the same parameters with all of the images to undistort them, so long as they were taken with the same camera. An undistorted road image is shown below:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, and used region of interest techniques from project 1 to get binary images of the lane lines (thresholding steps in the 5th cell of the IPython notebook). I used Sobel's method to find lines which had a large vertical gradient and converted to the HLS color space to threshold based on higher (but not too high) saturation values. Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform can be seen in the 6th cell of the iPython notebook. The `top_down_perspective()` function returns a transformed image that was warped using the matrix returned by `cv2.getPerspectiveTransform()` when passed source and destination points. The matrix returned is the transformation matrix necessary to move the points from source to the destination. The source and destination points are shown below:

```python
src = np.float32([(177, 720), (1132, 720), (593, 449), (683, 449)])
dst = np.float32([(320, 720), (960, 720), (320, 0), (960, 0)])
```

I verified that my perspective transform was working as expected by testing on images defined to have straight lines given in the set of example images.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the 7th cell I fit the lanes with 2nd order polynomials using the function `find_lanes()`. I use the histogram and windowing method from the lesson. It starts with creating a histogram of the bottom half of a binary thresholded image. The two peaks of the histogram, one left and one right of the midpoint, are used to identify where the lane lines start. The window loop checks if there are greater than a minimum number of non-zero pixels in the window's area and recenters the window at the average of the non-zero coordinates if there are enough pixels. The window then moves up to the next horixontal slice of the image and repeats until the entire image has had a window passed over it and all of the pixel 'hot spots' have been collected. A line of best fit is found through all of these pixels using polyfit. In the image below, the lane lines found through windowing are shown in red and the lines from the polyfit are the left and right edges of the green surface, which is the predicted shape of the lane.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This code is located in the 8th cell of the IPython notebook. The radius of curvature is calculated using a known function `R = [((1 + [dy/dx]^2))^(3/2)]/(d^2y/dx^2)`. I used the first and second order polynomials to plug into this equation. The radius of curvature was calculated at the points of the curve at the bottom of the image. Pixel to meter conversion was computed according to the equations:

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
```
and the python implementation of the equation is below:

```python
left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
```

To compute the distance from center, the true lane width in pixels was found by finding the centerpoint between the two bottom points of the line of best fit which was used to compute the conversion factor to meters, and then the difference between the center of the lane and the center of the images was calculated then converted to meters. The code went as follows:

```python
# Distance from center
lane_width_px = (right_fit[-1] - left_fit[-1])
meter_conversion = 3.7/lane_width_px
lane_center_px = left_fit[-1] + lane_width_px/2
car_center_px = out_img.shape[1]/2
dist_from_center = (lane_center_px-car_center_px)*meter_conversion # Positive: car left of center, Negative: car right of center
```

These calculations were included into the function `draw_lane_info()`, which uses `cv2.putText()` to draw this infomation onto the top left corner of the image. The upper left value is the left lane radius of curvature, the upper right value is the right lane radius of curvature, and the bottom value is the distance left of center. An example can be seen below:

![alt text][image6]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The full lane finding pipeline was tested in the 9th cell of the notebook using the function `draw_lanes()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are several ways I got the pipeline to work to get rid of miscalculations in finding the lane that make the pipeline less robust. I used the masking method from project 1 to get rid of the possiblity of the wall on the left of the lane interfering with lane calculations which was happening in a few frames. This is fine when driving on a single lane but if the car were to switch lanes, it would likely fail due to not having the full picture. I didn't use the method to find lines more easily on the fly by using the position of the previous lane either, which would have sped of some computation. Even with my method I did get wobbly lines and the line of best fit nearly failed in one frame due to low contrast or shadows. Tweaking the threshold values further might fix this. Another place tihs pipeline would definitely fail is if the lane lines weren't present or were poorly drawn. It would be great to identify all possible lanes in the feild of view of the car and then compute which lane the car is closest to being in instead of expecting the lane to be directly in front of the car at all times. This would make the pipeline able to identify lanes which switching lanes.
