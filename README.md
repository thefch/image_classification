# ComputerVision_cw2


---
# **STEPS for SIFT extraction**

### Step 1: Obtain the set of bag of features 
`build_vocabulary.m`
1. get images
2. extract sift features from images
3. get descriptors from extracted features
4. cluster the descriptors
    > will find similar features in each image and create visual words for each of it
5. obtain dictionary with visual words.

### Step 2: Obtain the bag of features for an image
`build_bag_of_sifts.m`
1. extract sift features of the image
2. get the descriptor for each point 
3. match the feature descriptors with the vocabulary of visual words (vocab.mat)
4. build the histogram with the features descriptors
	> it will be created with the frequency of each feature in an image
	each feature will correspond to a visual word in the dictionary
5. the visual words with the highest frequency will is the class of that image (prediction) 

> **visual words** -> a set of numbers representing a feature 

### Step 3: Classification

---


* useful lecture: https://youtu.be/iGZpJZhqEME
