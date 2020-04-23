function image_feats = fisher_encoding(image_paths,VOCAB_SIZE,FEATURE_STEP_SIZE,COLOUR_SPACE,BIN_SIZE)

% Load vocabulary
load('vocab_fisher.mat')
% vocab_size = size(means, 2);

% SIFT step size
STEP_SIZE = 5;
N = size(image_paths, 1);
image_feats = zeros(N, 2*VOCAB_SIZE*128);
for ii = 1:N
    I = imread(image_paths{ii});
    
    switch(COLOUR_SPACE)
        case "GRAYSCALE"       
            img = rgb2gray(I);
            img = im2single(img);
    end
    
    % For each image, get SIFT features with smaller step size
    [~, feats] = vl_dsift(img, 'Step', FEATURE_STEP_SIZE, 'Fast');
    feats_single = single(feats);
    
    image_feats(ii,:) = vl_fisher(feats_single , means, covariances, priors);
end


end