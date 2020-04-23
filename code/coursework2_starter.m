% Michal Mackiewicz, UEA
% This code has been adapted from the code 
% prepared by James Hays, Brown University

%% Step 0: Set up parameters, vlfeat, category list, and image paths.

FEATURE = 'fisher encoding';%'fisher encoding','colour histogram', 'tiny image','bag of sift', 'spatial pyramids'
IMAGE_SIZE =7;
COLOUR_SPACE = "GRAYSCALE";%RGB,HSV, GRAYSCALE, YCBR, NTSC
QUANTISATION = 8; %16,32
USE_NORM =  false;%normalization
USE_MEAN = true; %Standardization 
%DISTANCE = 'euclidean'; % cityblock,minkowski,L1,chisq,euclidean
DISTANCE = 'L1'; %L1,chisq need double quotes
% DISTANCE = 'cityblock';
STEP_SIZE = 5;
BIN_SIZE=6;
FEATURE_STEP_SIZE = 5;
VOCAB_SIZE = 200; % you need to test the influence of this parameter
MAX_LEVEL = 3;
CLASSIFIER = 'support vector machine' ;% 'nearest neighbor','support vector machine';
k=8;
LAMBDA =  0.00001; %0.0001

% Set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
run('vlfeat/toolbox/vl_setup');



data_path = '../data/';

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'House', ...
       'Industrial', 'Stadium', 'Underwater', 'TallBuilding', 'Street', ...
       'Highway', 'Field', 'Coast', 'Mountain', 'Forest'};
   
%This list of shortened category names is used later for visualization.
abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Hou', 'Ind', 'Sta', ...
    'Und', 'Bld', 'Str', 'HW', 'Fld', 'Cst', 'Mnt', 'For'};
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 100; 

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, test_image_paths, train_labels, test_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);
%   train_image_paths  1500x1   cell      
%   test_image_paths   1500x1   cell           
%   train_labels       1500x1   cell         
%   test_labels        1500x1   cell          




%% Step 1: Represent each image with the appropriate feature
% Each function to construct features should return an N x d matrix, where
% N is the number of paths passed to the function and d is the 
% dimensionality of each image representation. See the starter code for
% each function for more details.
fprintf('Using %s representation for images\n', FEATURE);


switch lower(FEATURE)    
    case 'tiny image'
        %You need to reimplement get_tiny_images. Allow function to take
        %parameters e.g. feature size.
        
        % image_paths is an N x 1 cell array of strings where each string is an
        %  image path on the file system.
        % image_feats is an N x d matrix of resized and then vectorized tiny
        %  images. E.g. if the images are resized to 16x16, d would equal 256.
        
        % To build a tiny image feature, simply resize the original image to a very
        % small square resolution, e.g. 16x16. You can either resize the images to
        % square while ignoring their aspect ratio or you can crop the center
        % square portion out of each image. Making the tiny images zero mean and
        % unit length (normalizing them) will increase performance modestly.
        
        train_image_feats = get_tiny_images(IMAGE_SIZE, train_image_paths,COLOUR_SPACE,USE_NORM,USE_MEAN);
        test_image_feats  = get_tiny_images(IMAGE_SIZE, test_image_paths,COLOUR_SPACE,USE_NORM,USE_MEAN);
    case 'colour histogram'
        %You should allow get_colour_histograms to take parameters e.g.
        %quantisation, colour space etc.
        train_image_feats = get_colour_histograms(QUANTISATION,train_image_paths,COLOUR_SPACE,USE_NORM,USE_MEAN);
        test_image_feats  = get_colour_histograms(QUANTISATION,test_image_paths,COLOUR_SPACE,USE_NORM,USE_MEAN);
     case 'bag of sift'
        % YOU CODE build_vocabulary.m
        if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
                       
            vocab = build_vocabulary(train_image_paths, VOCAB_SIZE,STEP_SIZE,COLOUR_SPACE,BIN_SIZE); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
        end
        
        % YOU CODE get_bags_of_sifts.m
        if ~exist('image_feats_sift.mat', 'file')
            train_image_feats = get_bags_of_sifts(train_image_paths,COLOUR_SPACE,DISTANCE,BIN_SIZE); %Allow for different sift parameters
            test_image_feats  = get_bags_of_sifts(test_image_paths, COLOUR_SPACE,DISTANCE,BIN_SIZE); 
            save('image_feats_sift.mat', 'train_image_feats', 'test_image_feats')
        else
            load('image_feats_sift.mat');
        end
      case 'spatial pyramids'
          if ~exist('vocab.mat', 'file')
            fprintf('No existing dictionary found. Computing one from training images\n')
                       
            vocab = build_vocabulary(train_image_paths, VOCAB_SIZE,STEP_SIZE,COLOUR_SPACE,BIN_SIZE); %Also allow for different sift parameters
            save('vocab.mat', 'vocab')
          end
          
          if ~exist('image_feats_sp.mat', 'file')
                        
               train_image_feats = spatial_pyramid(train_image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,FEATURE_STEP_SIZE,COLOUR_SPACE,BIN_SIZE);
               test_image_feats = spatial_pyramid(test_image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,FEATURE_STEP_SIZE,COLOUR_SPACE,BIN_SIZE);

              save('image_feats_sp.mat', 'train_image_feats', 'test_image_feats')
          else
            load('image_feats_sp.mat');
          end
        case 'fisher encoding'
            if ~exist('vocab_fisher.mat', 'file')
                fprintf('No existing dictionary found. Computing one from training images\n')

                [means, covariances, priors] = build_vocab_fisher(train_image_paths,VOCAB_SIZE,STEP_SIZE,COLOUR_SPACE);
                save('vocab_fisher.mat', 'means','covariances','priors')
            end
            
            if ~exist('image_feats_fisher.mat', 'file')

                train_image_feats = fisher_encoding(train_image_paths,VOCAB_SIZE,FEATURE_STEP_SIZE,COLOUR_SPACE,BIN_SIZE);
                test_image_feats = fisher_encoding(test_image_paths,VOCAB_SIZE,FEATURE_STEP_SIZE,COLOUR_SPACE,BIN_SIZE);

                save('image_feats_fisher.mat', 'train_image_feats', 'test_image_feats')
            else
                load('image_feats_fisher.mat');
            end
            
            
          % YOU CODE spatial pyramids method
end
%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.


fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER);
switch lower(CLASSIFIER)    
    case 'nearest neighbor'
        fprintf("-- k = %d \n",k);
    %Here, you need to reimplement nearest_neighbor_classify. My P-code
    %implementation has k=1 set. You need to allow for varying this
    %parameter.
        
    %This function will predict the category for every test image by finding
    %the training image with most similar features. Instead of 1 nearest
    %neighbor, you can vote based on k nearest neighbors which will increase
    %performance (although you need to pick a reasonable value for k).
    
    % image_feats is an N x d matrix, where d is the dimensionality of the
    %  feature representation.
    % train_labels is an N x 1 cell array, where each entry is a string
    %  indicating the ground truth category for each training image.
    % test_image_feats is an M x d matrix, where d is the dimensionality of the
    %  feature representation. You can assume M = N unless you've modified the
    %  starter code.
    % predicted_categories is an M x 1 cell array, where each entry is a string
    %  indicating the predicted category for each test image.
    % Useful functions: pdist2 (Matlab) and vl_alldist2 (from vlFeat toolbox)
        predicted_categories = nearest_neighbor_classify(k,train_image_feats, train_labels, test_image_feats,DISTANCE);
    case 'support vector machine'
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats,LAMBDA);
end

%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section. 

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?

create_results_webpage( train_image_paths, ...
                        test_image_paths, ...
                        train_labels, ...
                        test_labels, ...
                        categories, ...
                        abbr_categories, ...
                        predicted_categories)

                    %%
test(train_image_paths,test_image_paths,train_labels,categories,test_labels);
                    %%
if (CLASSIFIER == "support vector machine")
lbs  = [0.1,0.01,0.001,0.0001,0.00001];

for i=1:size(lbs,2)
    LAMBDA=lbs(:,i);
    fprintf("LAMBDA = %f \n",LAMBDA);
    predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats,LAMBDA);
    acc = get_accuracy(test_labels, categories,predicted_categories);
    fprintf("accuracy = %f \n\n",acc);
end
elseif(CLASSIFIER == "nearest neighbor")
    for i=1:15
        fprintf("k = %f \n",i);
         predicted_categories = nearest_neighbor_classify(k,train_image_feats, train_labels, test_image_feats,DISTANCE);
        acc = get_accuracy(test_labels, categories,predicted_categories);
        fprintf("accuracy = %f \n\n",acc);
    end
end
function accuracy = get_accuracy( test_labels, categories,predicted_categories)
    num_categories = length(categories);    
    confusion_matrix = zeros(num_categories, num_categories);
    for i=1:length(predicted_categories)
        row = find(strcmp(test_labels{i}, categories));
        column = find(strcmp(predicted_categories{i}, categories));
        confusion_matrix(row, column) = confusion_matrix(row, column) + 1;
    end
    %if the number of training examples and test casees are not equal, this
    %statement will be invalid.
    num_test_per_cat = length(test_labels) / num_categories;
    confusion_matrix = confusion_matrix ./ num_test_per_cat;   
    accuracy = mean(diag(confusion_matrix));
end

%%

