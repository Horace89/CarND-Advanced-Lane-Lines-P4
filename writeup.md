## Writeup


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

[calibr]: ./output_images/calibr.png "calibration"
[undistorted]: ./output_images/undistorted.png "undistorted"
[birdseye]: ./output_images/birds_eye_view.png "birdseye"
[white_mask]: ./output_images/white_mask.png "white_mask"
[yellow_mask]: ./output_images/yellow_mask.png "yellow_mask"
[combined_color_mask]: ./output_images/combined_color_mask.png "combined_color_mask"

[sobel_LandS]: ./output_images/sobel_LandS.png "sobel_LandS"
[sobel_combined]: ./output_images/sobel_combined.png "sobel_combined"
[color_sobel_combined]: ./output_images/color_sobel_combined.png "color_sobel_combined"
[window_fitting]: ./output_images/window_fitting.png "window_fitting"
[poly_fit]: ./output_images/poly_fit.png "poly_fit"
[lanes_on_orig]: ./output_images/lanes_on_orig.png "lanes_on_orig"
[complete_pipeline]: ./output_images/complete_pipeline.png "complete_pipeline"
[final_pipeline]: ./output_images/final_pipeline.png "final_pipeline"




## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points


---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()`. In the notebook, a couple examples of undistorted calibration images are shown.
![alt text][calibr]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once this calibration step is done, I show a couple of example real-world images after they are undistorted.
![alt text][undistorted]


#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


The next step is to compute a perspective transorm of the real-world pictures, so that we get a birds-eye view.

We first compute the coordinates of points of interest for a trapezoidal shape: top, bottom, center, left center, etc.:
```python
ytop = int(img_size[0] * .66)    #top
ybottom = img_size[0]            #bottom
width_div2 = img_size[1]//2     #width divided by 2
xc_window = width_div2          #center
xtopleft = xc_window - .23 * width_div2
xtopright = xc_window + .23 * width_div2
xbottomleft = xc_window - .96 * width_div2
xbottomright = xc_window + .96 * width_div2

src = np.float32([ [xbottomleft, ybottom],
                  [xbottomright, ybottom],
                  [xtopright, ytop],
                  [xtopleft, ytop] ])

dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],
                  [img_size[1],0],[0,0]])

M    = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```
and we compute the perspective transform matrix via the `cv2.getPerspectiveTransform` function (see notebook).



The final code for the perpective transform is then included in the `do_perspective_transform` helper function. We can see the result of this transform on an example image.

![alt text][birdseye]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image

First we look for good color masks to identify yellow and white lane markings in the birds-eye view image. We'll work in the HSV space via the `image_HSV = cv2.cvtColor(warped, cv2.COLOR_RGB2HSV)` transform. We'll then define min and max thresholds in this space. These thresholds were settled via a trial and error process. We get a yellow mask and a white mask in this way, and then we combine them.

![alt text][white_mask]
![alt text][yellow_mask]
![alt text][combined_color_mask]

Second we look for Sobel filters to the L and S channels of the image. I chose L and S channels because the lanes were more visible in L and S channels.

![alt text][sobel_LandS]

We then combine the Sobel filters:
![alt text][sobel_combined]

... and we add  them to the previously found color filter.
![alt text][color_sobel_combined]

This will be our final filter!



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I identify the lane pixels thanks to the convolution method shown in lesson 34. I copy-pasted its code then heavily refactored it. It is in the `find_window_centroids` function.
- The minimum level of convolution `MIN_CONV` for each level was defined by trial and error.
- A maximum drift threshold was also defined (see `MAX_DIFF` use)

![alt text][window_fitting]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After we get, for ex, the left lane picture as the result of the `fit_lines_full_search` function, we are able to fit a degree-2 polynom thanks to the code:
```python
vals = np.argwhere(img_L==1)
x_coords, y_coords = np.split(vals, 2, axis=1)
x_coords, y_coords = x_coords.squeeze(), y_coords.squeeze()
left_fit = np.polyfit(x_coords, y_coords, 2)
left_y = np.arange(11)*img_size[0]/10
left_fitx = left_fit[0]*left_y**2 + left_fit[1]*left_y + left_fit[2]
```
We get the following result:
![alt text][poly_fit]


I then compute the curvature and center with
```python
def get_curvature(y_eval, left_fit, right_fit):
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    img_height = 720
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    center = (((left_fit[0]*img_height**2+left_fit[1]*img_height+left_fit[2]) +(right_fit[0]*img_height**2+right_fit[1]*img_height+right_fit[2]) ) /2 - 640)*xm_per_pix
    return left_curverad, right_curverad, center
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After combining all these steps, we reproject the result back on to this original picture (see notebook).

![alt text][lanes_on_orig]



#### 7. Integration in a global pipeline

All the previous steps ware finally integrated in a complete Python class : see `Pipeline` which uses the `Line` class as suggested in the lesson.


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).


My final video is `output_project_video.mp4`. It includes a side display with the birds-eye view and the fitted lines.
Here's a [link to my video result](./output_project_video.mp4)

![alt text][final_pipeline]

---

### Discussion


My approach for the pipeline was :
- lane masking with color and Sobel filters
- lane isolation via a convolution window
- polynom fitting on the detected pixels
- sanity checks that reject anomalies among computed lans

This project was very long, with tedious trial and errors and debugging. The settings of the various hard-coded parameters was very long.

I could improve further the pipeline, which for now takes into account only one previous reference frame for its sanity check. It could for example use N=5 last reference lanes and use an algorithm like RANSAC for outliers detection.
