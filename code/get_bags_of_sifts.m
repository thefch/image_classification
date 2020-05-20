% % Implementated according to the starter code prepared by James Hays, Brown University
% % Michal Mackiewicz, UEA

 function image_feats = get_bags_of_sifts(image_paths,COLOR_SPACE,DISTANCE,BIN_SIZE,FEATURE_STEP_SIZE,USE_PHOW,USE_NORM)
% % image_paths is an N x 1 cell array of strings where each string is an
% % image path on the file system.
% 
% % This function assumes that 'vocab.mat' exists and contains an N x 128
% % matrix 'vocab' where each row is a kmeans centroid or a visual word. This
% % matrix is saved to disk rather than passed in a parameter to avoid
% % recomputing the vocabulary every time at significant expense.
% 
% % image_feats is an N x d matrix, where d is the dimensionality of the
% % feature representation. In this case, d will equal the number of clusters
% % or equivalently the number of entries in each image's histogram.
% 
% % You will want to construct SIFT features here in the same way you
% % did in build_vocabulary.m (except for possibly changing the sampling
% % rate) and then assign each local feature to its nearest cluster center
% % and build a histogram indicating how many times each cluster was used.
% % Don't forget to normalize the histogram, or else a larger image with more
% % SIFT features will look very different from a smaller version of the same
% % image.
% 
% %{
% Useful functions:
% [locations, SIFT_features] = vl_dsift(img) 
%  http://www.vlfeat.org/matlab/vl_dsift.html
%  locations is a 2 x n list list of locations, which can be used for extra
%   credit if you are constructing a "spatial pyramid".
%  SIFT_features is a 128 x N matrix of SIFT features
%   note: there are step, bin size, and smoothing parameters you can
%   manipulate for vl_dsift(). We recommend debugging with the 'fast'
%   parameter. This approximate version of SIFT is about 20 times faster to
%   compute. Also, be sure not to use the default value of step size. It will
%   be very slow and you'll see relatively little performance gain from
%   extremely dense sampling. You are welcome to use your own SIFT feature
%   code! It will probably be slower, though.
% 
% D = vl_alldist2(X,Y) 
%    http://www.vlfeat.org/matlab/vl_alldist2.html
%     returns the pairwise distance matrix D of the columns of X and Y. 
%     D(i,j) = sum (X(:,i) - Y(:,j)).^2
%     Note that vl_feat represents points as columns vs this code (and Matlab
%     in general) represents points as rows. So you probably want to use the
%     transpose operator '  You can use this to figure out the closest
%     cluster center for every SIFT feature. You could easily code this
%     yourself, but vl_alldist2 tends to be much faster.
% %}
% 
% % try histogram normalisation
% % imgHistNorm = imgHist./size(SIFT_features,2);

load('vocab.mat');

vocab_size = size(vocab, 1);
N = size(image_paths, 1);
image_feats = zeros(N, vocab_size);
USE_MEAN = false;
% MEAN is not used, no improvement in accuracy
for ii=1:N
    % for debugging
%     if (mod(ii,200)==0)
% %         fprintf(" %d \n",ii);
%     end
    I =imread(image_paths{ii});
    
    % extract sift features
    if(USE_PHOW)
       [~, features]= get_phow(I,COLOR_SPACE,FEATURE_STEP_SIZE);
    else
        if(COLOR_SPACE == "GRAYSCALE")
            img = im2single(rgb2gray(I));
        else
            img = get_color_values(I,COLOR_SPACE);
        end
        [~, features] = vl_dsift(img,'Step',FEATURE_STEP_SIZE);
        features = single(features);        
    end      
    
     dists = vl_alldist2(features,vocab',DISTANCE);    
     [~, inds] = min(dists, [], 2);
%  for debugging
%     it does the same as alldist, but much faster, it uses a kd-tree
%     [inds, ~] = knnsearch(vocab, features', 'K', 1);
    
    % construct the histogram(sift counts) of with the current sift features
    hist = histc(inds, 1:vocab_size)';
    
    if(USE_MEAN)
        hist = hist / mean(hist(:));
    end
    
    % add the histogram to the feature list
    % each row will represent a histogtram of sift counts of each image
    image_feats(ii,:) = hist;

end
 end

