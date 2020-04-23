function [means, covariances, priors] = build_vocab_fisher(image_paths,VOCAB_SIZE,STEP_SIZE,COLORSPACE)
% SIFT step size
STEP_SIZE = 75;

% Get SIFT features for each image
feats = cell(1, size(image_paths, 1));
N = size(image_paths, 1);
for i = 1:N
    I = imread(image_paths{i});
    switch(COLORSPACE)
        case 'GRAYSCALE'
            img = rgb2gray(I);
            img = im2single(img);
    end
    
    [~, feats{1,i}] = vl_dsift(img, 'Step', STEP_SIZE, 'Fast');
end

% Use GMM as vocabulary
[means, covariances, priors] = vl_gmm(single(cell2mat(feats)), VOCAB_SIZE);
end