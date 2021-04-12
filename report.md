

**Deliverable #2** - https://github.com/KevinHuang8/caltech-ee148-spring2020-hw02

- Matched filtering algorithm: `run_predictions.py`
- Generate PR curves: `eval_detector.py`

**Deliverable #1**

1. The algorithm is as follows (Note that this is very similar to the main algorithm I used for homework 1):

   1. Let the kernel $k$ (also known as the template) be an arbitrary image of a red light. 
   2. For each image, perform z-score normalization sample-wise. Normalize the kernel as well.
   3. Define a set of scales $s = [s_1, s_2, \dots, s_n]$, where $0 < s_i \leq 1$.
   4. Repeat steps 5 - 12 for $i = 1,\dots, n$:
   5. Scale the kernel by a factor of $s_i$ 
   6. Convolve the image $I$ with $k$, to produce an output $r$.
   7. Take the $\alpha$, $\alpha \in (0, 1)$, percentile of pixels in $r$ to be the candidate locations for a red light. Store this in a new matrix $r'$, and set all other pixels to 0 in $r'$.
   8. Cluster the nonzero pixels in $r'$ by labeling all contiguous nonzero pixels by unique identifier. This is done by a simple flood fill algorithm.
   9. Compute the centers of each cluster in $r'$, which are chosen as the locations of each traffic light. The top left corner of the associated bounding box of the cluster is the center of the cluster. The bounding boxes sizes are the size of the kernel, but with twice the height. The height is due to the fact that the kernel is designed to be just the upper half of a red light. 
   10. For each computed cluster in $r'$, calculate the *eccentricity* of the cluster, defined as the span of the pixels in cluster in the $x$ dimension over the span of the pixels in the cluster in the $y$ dimension, or vice versa to make the ratio > $1$. If the eccentricity is greater than 1.5, discard the cluster. The reasoning behind this step is that we know that traffic lights are circular, so if there is a highly elongated shape detected, it is almost certainly not a traffic light.
   11. For each computed cluster in $r'$, if there are more pixels in that cluster than the kernel itself, then discard that cluster. This is because we cannot reasonably detect objects larger than our kernel using this algorithm, so any such detections are almost certainly false positives.
   12. Let $|k|^2$ be the dot product of $k$ with itself. If there is a perfect match with the kernel, then there should be a pixel in $r$ with a value of $|k|^2$. Thus, the confidence score of each bounding box is calculated as follows:
       - Let $m = \max_{i \in C} i$, where $C$ is the cluster of pixels associated with the bounding box
       - The confidence is given by $\exp\left(-\left(\frac{m - |k|^2}{0.5|k|^2}\right)^2\right)$. The interpretation is that the closer the max value in the cluster in $r'$ associated with the bounding box is to the theoretical true value, the closer the confidence is to $1$, and vice versa for $0$. 
   13. Now, combine all bounding boxes obtained from all kernel scales $s$. 
   14. For any overlapping bounding boxes, choose the one with the highest confidence and discard the rest.

   Parameter values of $\alpha = 0.9995$ and $s = [1, \frac12, \frac13]$ were chosen for this assignment (fairly arbitrarily, little tuning was done). 

2. Template:

   <img src="C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\filter2.png" alt="filter2" style="zoom:150%;" />

   Heatmap visualization:

   

   Small Scale:

   ![heatmap_small](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\heatmap_small.png)

   Medium Scale:

   ![heatmap_med](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\heatmap_med.png)

   Large Scale:

   ![heatmap_large](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\heatmap_large.png)

   The above images show the heatmap generated when the template was convolved with the image, with different size templates. We see that a larger template detects larger and coarser features, while a smaller template detects smaller and finer features. The bounding box locations were localized by taking the top $\alpha$ percentile of a heatmap, clustering the resulting pixels, and then taking the centers of those clusters (described in more detailed in the algorithm description above). 

3. Ground truth - left; predictions - right

   "RL-010.jpg"

   <p float="left">
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411175613309.png" width="480" />
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411175659225.png" width="480" /> 
   </p>

   "RL-325.jpg"

   <p float="left">
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411175817732.png" width="480" />
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411175837660.png" width="480" /> 
   </p>

   "RL-012.jpg"

   <p float="left">
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411180016345.png" width="480" />
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411180033314.png" width="480" /> 
   </p>

   In the first example, the detections with the highest confidences are the 3 red traffic lights, so the algorithm should have perfect precision and recall on this image if the confidence threshold is set high enough. The algorithm does detect many other objects that are not traffic lights, but these have lower confidences, as we would expect. This example works well because the traffic lights are very prominent and large, with few other distractions or other objects that might resemble red lights.

   In the second example, there are no red lights in the ground truth. Indeed, the algorithm correctly does not detect any red lights with high confidence. Though there are numerous false detections, all of the detections have very low confidence, as we would expect. This works well because there are essentially no red objects in the picture, so nothing will really match very well with the red light template.

   The last example is similar to the first, where the algorithm detects all of the red lights with high confidence, and it is not too fooled by the other red objects in the picture, as they are identified with fairly low confidence. Ideally, there would be no false detections at all, but the confidence level is behaving generally as we would expect it to here. Again, this image works well because the traffic lights are all big and prominent, and the other red objects do not exactly fit the template very well (they are either very small, big, or are irregularly shaped, so the algorithm knows that they are not red lights fairly confidently).

4. 

   "RL-333.jpg"

   <p float="left">
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411180223809.png" width="480" />
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411180210929.png" width="480" /> 
   </p>

   "RL-028.jpg"

   <p float="left">
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411180414413.png" width="480" />
     <img src="C:\Users\kehua\AppData\Roaming\Typora\typora-user-images\image-20210411180424391.png" width="480" /> 
   </p>

   In the first image above, we see that the algorithm was confused by the multitude of other light sources and reflections. While the algorithm did correctly identify all red traffic lights with high confidence, it also identified non-traffic lights with high confidence, such as the reflections of the traffic lights on the hood of the car. This demonstrates how it can be very hard for the algorithm to distinguish between actual red lights and reflections of red lights.

   In the second image, the algorithm is confused by all of the tail lights on the cars, which are at the right distance away and shape to be very similar to red lights. This reflects a weakness in the algorithm, as the algorithm does not take into account the surrounding context of the template. Thus, red car lights and red traffic lights look nearly identical if the car lights happen to be the right shape, as they are in this example above.

5. Main Algorithm

   ![PR_train](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\PR_train.png)

   ![PR_test](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\PR_test.png)

   For our particular algorithm, in all PR curves we see that decreasing the IOU threshold increases the performance of our algorithm in both precision and recall. The reasoning is likely because the algorithm's detection locations don't perfectly match the ground truth, but are close enough, so a more stringent IOU threshold will discount true positives that are close enough to the ground truth but don't match well enough.

   We see that the algorithm performs better on the test set than the training set! That is, the test PR curves are mostly above the training PR curves. This is at first glance strange, since we would expect the opposite. However, it is important to note that I did not really tune the parameters of the algorithm on the training set much. I essentially just chose the parameters $\alpha$ and $s$ looking at one or two images from the training set, and didn't attempt to optimize them according to the training set. That is, I didn't really "train" this algorithm at all, making the distinction between the training and test sets not really matter. The test set happens to contain many images with no ground truth objects at all, and the algorithm generally performs well on those images. Thus, the difference in performance between the two datasets is mostly due to random chance. 

   If I did indeed optimize this algorithm more using the training set, then we would expect the opposite result, where the algorithm performs better on the training set than the test set.

   Weakened Algorithm:

   ![PR_train_weak](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\PR_train_weak.png)

   ![PR_test_weak](C:\Users\kehua\Documents\Caltech\CS 148\caltech-ee148-spring2020-hw02\PR_test_weak.png)

   The weakened algorithm simply removes step 10 from the algorithm, the step that filters by eccentricity. We see that the weakened algorithm performs very slightly worse than the standard algorithm, but not by much. For example, we see that the max precision of the weakened algorithm is about 0.14 while the max precision of the standard is about 0.175 for the training set. The effect is not very large, meaning that the eccentricity calculation does not have too much of an effect, though it does make a difference. 

   One use of comparing with a less capable version of the algorithm is determining the effect of changes to the algorithm on performance. We can quantify how much a specific change improves our algorithm.