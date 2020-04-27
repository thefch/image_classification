# ComputerVision_cw2

# **Steps for SIFT extraction**

### Step 1: Obtain the set of bag of features 
`build_vocabulary.m`
1. get images
2. extract sift features from images
3. get descriptors from extracted features
4. cluster the descriptors
    > will find similar features in each image and create visual words for each of it
    
5. obtain dictionary with visual words.

### Step 2: Obtain the bag of features for an image
`get_bag_of_sifts.m`
1. extract sift features of the image
2. get the descriptor for each point 
3. match the feature descriptors with the vocabulary of visual words (vocab.mat)
4. build the histogram with the features descriptors
	> it will be created with the frequency of each feature in an image
	each feature will correspond to a visual word in the dictionary
	
5. the visual words with the highest frequency will is the class of that image (prediction) 

> **visual words** -> a set of numbers representing a feature 


# **Steps for Spatial Pyramid**
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




---
# **Steps for Classification**

### 1. kNN

### 2. SVM



---
* useful lecture: https://youtu.be/iGZpJZhqEME
