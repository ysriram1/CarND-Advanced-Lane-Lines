## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*Please note that this report has been written by using the provided writeup_template.md file*

In this project, we are tasked with creating a pipeline that takes in a video of a car driving on a highway and outputs a video with the car's lane highlighted.

The specific goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/calibration1.png "calibration1"
[image2]: ./writeup_images/calibration2.png "calibration2"
[image3]: ./writeup_images/sharp.png "sharpened"
[image4]: ./writeup_images/warp.png "warp"
[image5]: ./writeup_images/left.png "left"
[image6]: ./writeup_images/right.png "right"
[image7]: ./writeup_images/combined.png "combined"
[image8]: ./writeup_images/hist.png "hist"
[image9]: ./writeup_images/windows.png "windows"
[image10]: ./writeup_images/lines.png "lines"
[image11]: ./writeup_images/final.png "final"



## Files, System Configuration, and Dependencies

### Files

- *README.md* (this file) is the project report
- *code_clean.ipynb* contains the image processing pipeline
- *results/lane_lines_video.mp4* is a video of the project video annotated with lane line

### System Configuration

- Windows 10
- Nvidia GeForce GTX 1070
- Intel i7 4.20GHz
- 32GB RAM

### Python Package Dependencies

- Python 3.6
- moviepy
- opencv
- matplotlib
- numpy
- os
- itertools


## Camera Calibration

Camera calibration is a processing of un-doing the distortion that is built into images captured by a camera. Different cameras distort images differently. Hence, it is important to first find the distortion matrix of the camera used to capture the video in the project and undistort before proceeding with the rest of the lane detection pipeline.

We start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here we are assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time we successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

We then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  We applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result when applied to a photo of a chess board:

![][image1]

## Lane Detection Pipeline

### Undistortion and Sharpening

*Code: Look at Camera Calibration section, `undistort_img()`, and `sharpen()`*

Now that we have a method of undistorting images, we proceed to apply this function to the highway images. Here is the result when applied to a sample highway image:

![][image2]

After undistorting the image, we also decided that it makes sense to sharpen the image. This would result in the edges (including the lane lines) appear more clearly in the image and any lane detection methods used down the line would benefit from this step. Image sharpening was performed by using a kernel matrix and applying it to the image using opencv's `filter()` function -- highpass filtering. The kernel matrix emphases the pixel rather than its surrounding pixels.

kernel used:
```
[[-1,-1,-1,-1,-1],
 [-1,2,2,2,-1],
 [-1,2,16,2,-1],
 [-1,2,2,2,-1],
 [-1,-1,-1,-1,-1]]) / 16.
```
And here is result of this step:

![][image3]

### Warping using Perspective Transform

*Code: `warp()`*

Applying perspective transform turned out be a critical step in this project. We were able to warp the image such that the lane lines as they should be if were viewing the road from the sky (see image below). This process also essentially *masked* the image so that we would just have to apply the remaining image processing steps to only the region with lane lines.

In order to perform a perspective transform, we need to provide the coordintes of the source points -- the 4 points that define the region with the lane lines in the original image -- and the destination points -- the 4 points that indicate where we want the original for points to appear in the warpped image. These points were fed into opencv's `getPerspectiveTransform()` to get the transformation matrix, which was applied to the image for transformation.

The source and destination points were ascertained via a process of trail and error and by visually inspecting the original, unwarpped image.

```
src = np.float32([[575,470],[750,470],[260,670],[1100,670]])
dst = np.float32([[200,0],[w-200,0],[200,h],[w-200,h]])
```
Here is the result of applying the perspective transform:

![][image4]

### Finding Lanes
*Code: `apply_HS_thresh()`, `apply_RGB_thresh()`, and `add_binary_images()`*

Since we have two lane lines -- one on the left and the other on the right. However, both these lanes are different from eachother; the left lane is yellow in color and continuous and the right lane is white and dashed. Hence, we ended up using two different color maps and approaches to detect each of the two lanes. Once detected the corresponding binary images simply combined to result in the final binary image.

To detect the left lane line, the HLS color channel was used. The following thresholds were used:

Hue - 20 to 75
Saturation - 100 to 255
Lightness - 0 to 255 (basically no thresholding)

These values were chosen using trail and error and an internet search on the ranges work well for yellow. These thresholds resulted in the following lane detection:

![][image5]

The right lane was a little harder to detect by just using the HLS color channel. It actually turned out that the RGB channel was able to detect the right lane. The Red, Green, and Blue channels had to be picked in such a way that a wide range of colors close to white would be chosen. This worked since the right lanes are white in color. We used the following thresholds:

Red - 190 to 255
Green - 190 to 255
Blue - 190 to 255

This resulted in the following binary image:

![][image6]

Both these lane images were combined to create a single binary image with both the lane lines detected.

![][image7]

One surprising result of this exercise was that using the sobel operator (gradient) did not actually produce good results. It was either blacking out most of the image, if the thresholds were narrow, or detecting too much noise, if the thresholds were wide.

### Drawing Lanes using a Sliding Windows

*Code: `sliding_windows()`, `new_lane_lines()`, and `draw_lane_lines()`*

Now that we had a binary image with the lane lines detected, it was time to draw the actual lane lines and ascertain their curvature by ignoring any noise surrounding the detections. In order to accomplish this, we used the sliding window technique explained in the lecture.

As a first step, we generated a historgram that shows the pixel intensities along the x-axis (horizontal) of the image. The peaks of the histogram are the points on the x-axis where the lanes are.

![][image8]

Now these two points on the x-axis are used to constructed sliding windows of a fixed height and width to detect the exact trajectory of the lanes. The pixels encapsulated in the windows are then used to fit a second order polynomial.

![][image9]

![][image10]


### Calculating Curvature

*Code: `add_curvature_info()`*

Using the coefficients of the polynomial line we fit to the lanes, we calculate the curvature of the two lanes. The following formula was used:

`curvature_left = ((1 + (2*fit[0]*y_ +
                    fit[1])**2)**1.5) / np.absolute(2*fit[0])`

Here, fit is a numpy array containing the coefficients of the second order polynomial applied to the valid pixels on each side.

### Final Output

*Code: `pipeline()`*

Now that we had the lane lines detected and added to the warpped image, we proceed to unwarpping the image and marking these lines on the original image. This is accomplished using the `M_inv` matrix, which transforms the dst points back into their original src locations in the original image.

The resulting image looks like the following:

![][image11]

### Conclusion

Overall this was challenging but rewarding project. The one area that took a long time was the lane line detection. Initially, we expected the thresholding the Sobel operator to do a good job of finding the lane lines, but it proved insufficient. In the final model, we ended up not using the sobel operator at all. The HLS color channels were used to detect the left lane and the RGB channels were used to detect the right lane. The sliding windows technique proved to be very efficient in finding the 'good' pixels and filtering out the noise (or the bad, non-lane line pixels). In addition, the perspective transform also proved useful as it acted as a mask and removed all other pixels except for those in the region with the lane lines.
