# Sift features

###### Scale - Invariant feature transform

* Invariant to transformation

* Features based on image's gradients

* It produces a dictionary of visual words

  * with size 128xN

    > N is the size of the dictionary.

  * Each word is a histogram of sift descriptors

    (eg. kitchen, store, etc.)

They were used for the feature extraction of <ins>Bag of Sift</ins> and <ins>Spatial Pyramid</ins>.

<img src="pages-assets\images\SIFTexplaining.png" alt="SIFTexplaining" style="zoom:50%;" />

<img src="pages-assets\images\siftExampleOnline.png" alt="siftExampleOnline" style="zoom:40%;" />

<img src="pages-assets\images\sift.png" alt="sift" style="zoom:70%;"  />

​                                         <em>SIFT features between 2 images with the same scene </em>



---

## Different color spaces

The color spaces with more than 1 channel had a different approach than the grayscale color:

1. Concatenate the color channels into a 2D matrix

2. Use vl_phow

   * Dense SIFT features

   * Extract SIFTs from all channels separately

     > Not much improvement with this method



<img src="presentation\colorChannels.png" alt="colorChannels" style="zoom:50%;"  />



## Spatial Pyramid

* Collection of orderless feature histograms
* Each level consists of a grid with histograms
* Histograms are created by the local SIFT descriptors on each quadrant
* For each level a weight is applied

<img src="pages-assets\images\spatial_pyramidHistExample.png" alt="spatial_pyramidHistExample" style="zoom:80%;"  />



## Additional scene recognition method: 

###                                  <ins>Fisher Encoding</ins>

 

<em>Introduced in Image Classification with the Fisher Vector: Theory and Practice (Perronnin et al. 2007)</em>

> Build upon the Bag Of Visual Words method

##### Fisher Vocabulary:

1. Extract SIFT descriptors from each image.
2. Apply Gaussian Mixture Model (GMM) to the obtained features.
   * Instead of clustering
   * returns the means, covariances, priors that are used as a vocabulary



##### Fisher Vector:

1. Extract SIFT descriptors 
2. Compute the fisher vector of each image by using their SIFT features and the already computed vocabulary
   * Each vector represents an image

![Fisher](pages-assets\images\Fisher.png)



##### Comparison with BoS:

<ins>Advantages:</ins>

* It can be computed with much smaller vocabularies

<ins>Disadvantages:</ins>

* Takes more storage
  * (2*D+1)*N –1



# Steps:

## Steps for SIFT extraction

### Step 1: Obtain the set of bag of features 
`build_vocabulary.m`
1. get images

2. extract sift features from images

3. get descriptors from extracted features

4. cluster the descriptors
   
    > will find similar features in each image and create visual words for each of it
    
5. obtain dictionary with visual words.

    

<img src="pages-assets\images\bagofwords.jpg" alt="SIFTexplaining" style="zoom:50%;" />

​                                        <em>Image illustrating the process of creating a vocabulary of visual words</em> 



### Step 2: Obtain the bag of features for an image

`get_bag_of_sifts.m`

1. extract sift features of the image
2. get the descriptor for each point 
3. match the feature descriptors with the vocabulary of visual words (vocab.mat)
4. build the histogram with the features descriptors
	> it will be created with the frequency of each feature in an image
	> each feature will correspond to a visual word in the dictionary
	
5. the visual words with the highest frequency will is the class of that image (prediction) 

> **visual words** -> a set of numbers representing a feature 

---

## Steps for Spatial Pyramid

`spatial_pyramid.m`
1. get images

2. extract sift features from images

3. get descriptors from extracted features

4. find the minimum distance of the the extracted features and the ons from the already computed vocabulary <br /> 
	`D = vl_alldist2(vocab',features)`<br /> 
	`[~,ind] = min(D);`.
	
5. construct a histogram with those values.
	
	> It will be the histogram with SIFT features for Level 0 of the pyramid.
	
6. Create a matrix with the total levels of the pyramid <br /> 
	6.1 Each level will have a number of quadrants<br /> 
	6.2 Each  quadrant will be represented with a histogram of its SIFT features.<br /> 
	6.3 Then each level will have those histograms concantated into a row, for the pyramid.<br />
	
	>In will result into a bigger histogram
	
7. Apply the appropriate weight to each level

   

<img src="pages-assets\images\spatial_pyramid.png" alt="spatial_pyramid" style="zoom:90%;" />

---



## Classification



#### 1. kNN
#### 2. SVM
---
* useful lecture: https://youtu.be/iGZpJZhqEME

---

---

# Results

## Bag of Sift

​																		 **kNN**

![knnBoSsift](pages-assets\graphs\knnBoSsift.PNG)



​																		 **SVM**

![svmBoSsift](pages-assets\graphs\svmBoSsift.PNG)

[RGB Confusion Matrix](pages-assets\html\bos_svm_rgb_vocab1000_ss200_fss5\index.html)

[GRAYSCALE Confusion Matrix](pages-assets\html\bos_svm_gray_vocab1000_ss10_fss5\index.html)													



![knnsvmBoSphow](pages-assets\graphs\knnsvmBoSphow.PNG)

[RGB Confusion Matrix](pages-assets\html\bos_phow1_knn_rgb_vocab1000_ss10_fss5\index.html)

[GRAYSCALE Confusion Matrix](pages-assets\html\bos_phow1_svm_rgb_vocab1000_ss10_fss5\index.html)		

---



## Spatial Pyramid

​																		 **kNN**

![knnSPsift](pages-assets\graphs\knnSPsift.PNG)

[RGB Confusion Matrix](pages-assets\html\sp_knn_rgb_vocab400_ss10_fss5_level2\index.html)

[GRAYSCALE Confusion Matrix](pages-assets\html\sp_knn_gray_vocab200_ss200_fss5_level2\index.html)



​																		 **SVM**

![svmSPsift](C:\Users\fanos\Desktop\University\Computer Vision\image_classification_cw\pages-assets\graphs\svmSPsift.PNG)

[RGB Confusion Matrix](pages-assets\html\sp_svm_rgb_vocab1000_ss10_fss5_level2\index.html)

[GRAYSCALE Confusion Matrix](pages-assets\html\sp_svm_gray_vocab1000_ss10_fss5_level3\index.html)

---



## Fisher Vector

​																		 **kNN**

![knnFISHER](pages-assets\graphs\knnFISHER.PNG)

[RGB Confusion Matrix](pages-assets\html\fisher_knn_rgb_vocab100_ss75_fs5\index.html)

[GRAYSCALE Confusion Matrix](pages-assets\html\fisher_knn_gray_vocab100_ss75_fs5\index.html)



​																		 **SVM**

![svmFISHER](pages-assets\graphs\svmFISHER.PNG)

[RGB Confusion Matrix](pages-assets\html\fisher_svm_rgb_vocab100_ss75_fs5\index.html)

[GRAYSCALE Confusion Matrix](pages-assets\html\fisher_svm_gray_vocab100_ss75_fs5\index.html)



---



# Conclusion

* The less the step size the slower and more memory MATLAB was using
* Spatial Pyramid gave good results till level 2 with RGB color space
  * After level 2, not really better results, much more computational power
* Feature Step Size of 5 seemed to worked fine with all methods
* Fisher Vector method worked better with smaller vocabulary
* kNN classifier was really slow in comparison with SVM 
  * Too much data