% % Implementated according to the starter code prepared by James Hays, Brown University
% % Michal Mackiewicz, UEA
% 

 function image_feats = get_bags_of_sifts(image_paths,COLOR_SPACE,DISTANCE,BIN_SIZE,FEATURE_STEP_SIZE,vocab)
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
% 
% load('vocab.mat');
% vocab_size = size(vocab, 1);
% N = size(image_paths, 1);
% 
% image_feats = zeros(N,vocab_size);
% 
% 
% for ii=1:N
%     if(mod(ii,200) == 0)
%        disp(ii); 
%     end
%     
%     I = imread(image_paths{ii});
% %    imshow(I);
%     switch(COLOR_SPACE)
%         case("GRAYSCALE")
%             img = rgb2gray(I);
%             img = im2single(img);
%         case("RGB")
%             img = get_color_values(I); 
%     end
%     
% 
% %     imshow(img);
%     
%     % DESCRIPTORS from sift features of the specific image
%     [~, feats] = vl_dsift(img,'Step',FEATURE_STEP_SIZE,'Fast') ;
%     
%     % create the histogram of visual words
%     hist = get_hist(feats, vocab,DISTANCE);
% 
% %     histNorm = hist - min(hist(:));
% %     histNorm = histNorm ./ max(histNorm(:));
% %     
% %     image_feats(ii,:) = hist;
%     % set the visual words that corresponds to the current image (row)
%     [image_feats(ii,:), ~] = histcounts(hist,vocab_size,'Normalization', 'probability');
% 
% end
% end
% 
% 
% 
% % create the histogram with the extracted featrues
% function hist = get_hist(feats, vocab,DISTANCE)
%      feats_single = single(feats);
%     dists = vl_alldist2(feats_single,vocab');    
%     [~, inds] = min(dists, [], 2);
%     hist = inds;
% end

load('vocab.mat');

vocab_size = size(vocab, 1);
N = size(image_paths, 1);
image_feats = zeros(N, vocab_size);
for ii=1:N
    if (mod(ii,200)==0)
        fprintf(" %d \n",ii);
    end
    if(COLOR_SPACE == "GRAYSCALE")
        img = im2single(rgb2gray(imread(image_paths{ii})));
    else
        img = get_color_values(imread(image_paths{ii}),COLOR_SPACE);
    end
    [~, features] = vl_dsift(img,'Step',FEATURE_STEP_SIZE);
    features = single(features);
    [inds2, ~] = knnsearch(vocab, features', 'K', 1);
%      dists = vl_alldist2(features,vocab',DISTANCE);    
%      [y, inds2] = min(dists, [], 2);
%     image_feats(ii,:) = histc(inds2, 1:vocab_size)';
%     image_feats(ii,:)= histcounts(inds2, vocab_size);
%     image_feats(ii,:)= histcounts2(inds2, vocab_size);
    image_feats(ii,:) = histc(inds2, 1:vocab_size)';

end
end
