% Based on James Hays, Brown University

%This function will train a linear SVM for every category (i.e. one vs all)
%and then use the learned linear classifiers to predict the category of
%every test image. Every test feature will be evaluated with all 15 SVMs
%and the most confident SVM will "win". Confidence, or distance from the
%margin, is W*X + B where '*' is the inner product or dot product and W and
%B are the learned hyperplane parameters. 

function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats,LAMBDA)
% image_feats is an N x d matrix, where d is the dimensionality of the
%  feature representation.
% train_labels is an N x 1 cell array, where each entry is a string
%  indicating the ground truth category for each training image.
% test_image_feats is an M x d matrix, where d is the dimensionality of the
%  feature representation. You can assume M = N unless you've modified the
%  starter code.
% predicted_categories is an M x 1 cell array, where each entry is a string
%  indicating the predicted category for each test image.

%{
Useful functions:
 matching_indices = strcmp(string, cell_array_of_strings)
 
  This can tell you which indices in train_labels match a particular
  category. This is useful for creating the binary labels for each SVM
  training task.

[W B] = vl_svmtrain(features, labels, LAMBDA)
  http://www.vlfeat.org/matlab/vl_svmtrain.html

  This function trains linear svms based on training examples, binary
  labels (-1 or 1), and LAMBDA which regularizes the linear classifier
  by encouraging W to be of small magnitude. LAMBDA is a very important
  parameter! You might need to experiment with a wide range of values for
  LAMBDA, e.g. 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10.

  Matlab has a built in SVM, see 'help svmtrain', which is more general,
  but it obfuscates the learned SVM parameters in the case of the linear
  model. This makes it hard to compute "confidences" which are needed for
  one-vs-all classification.

%}

%unique() is used to get the category list from the observed training
%category list. 'categories' will not be in the same order as in coursework_starter,
%because unique() sorts them. This shouldn't really matter, though.

categories = unique(train_labels); 
num_categories = length(categories);
num_train = size(train_image_feats, 1);

dim = size(test_image_feats, 2);
Ws = zeros(num_categories, dim);
Bs = zeros(num_categories, 1);
for ii=1:num_categories
    
    % set all the labels to -1
    labels = ones(num_train,1).*-1;
    
    % set the currect label to 1
    % it will distinguish the current image with the rest.
    % (this label) vs (the rest)
    labels(strcmp(categories{ii}, train_labels)) = 1;
    
    % use svm with specific LAMBDA value
    [W, B] = vl_svmtrain(train_image_feats', labels, LAMBDA, 'MaxNumIterations', 1e5);
    % the lambda valuw will determine the distance of the values that will
    % be considred when classifyng, the lower the further 
    
    % copy the weights and offset of each class
    Ws(ii,:) = W';
    Bs(ii) = B;
end

images_N = size(test_image_feats, 1);

confidences = Ws*test_image_feats'+repmat(Bs,1,images_N);
[~, indices] = max(confidences);
% indeices hold the classified class of each image
predicted_categories = categories(indices);
end