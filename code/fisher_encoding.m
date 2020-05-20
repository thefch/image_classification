function image_feats = fisher_encoding(image_paths,VOCAB_SIZE,FEATURE_STEP_SIZE,COLOUR_SPACE,BIN_SIZE)

% Load vocabulary
load('vocab_fisher.mat')
N = size(image_paths, 1);
image_feats = zeros(N, 2*VOCAB_SIZE*128);
for ii = 1:N
    I = imread(image_paths{ii});
    
    if(COLOUR_SPACE == "GRAYSCALE")
        img = im2single(rgb2gray(I));  
    else
        img = get_color_values(I,COLOUR_SPACE);
    end
    
%   extract sift features from the images
    [~, feats] = vl_dsift(img, 'Step', FEATURE_STEP_SIZE, 'Fast');
    feats_single = single(feats);
    
%   computer the fisher vector with from the image's sift features and the vocabulary     
    image_feats(ii,:) = vl_fisher(feats_single , means, covariances, priors);
end
end