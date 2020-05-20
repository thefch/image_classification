function [means, covariances, priors] = build_vocab_fisher(image_paths,VOCAB_SIZE,STEP_SIZE,COLORSPACE)

% Get SIFT features for each image
feats = cell(1, size(image_paths, 1));
N = size(image_paths, 1);
for i = 1:N
    I = imread(image_paths{i});
    if(COLORSPACE == "GRAYSCALE")
        img = im2single(rgb2gray(I));  
    else
        img = get_color_values(I,COLORSPACE);
    end
    
    [~, feats{1,i}] = vl_dsift(img, 'Step', STEP_SIZE, 'Fast');
end

% Use GMM as vocabulary
[means, covariances, priors] = vl_gmm(single(cell2mat(feats)), VOCAB_SIZE);
% means, covariances, priors will represent hthe volcabulary
end

%   extract the color values
%   extract sift descripros
%   apply gaussian mixture model (instead of clustering)