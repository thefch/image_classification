function image_feats = spatial_pyramid(image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,STEP_SIZE,COLORSPACE,BIN_SIZE,vocab)
%train_image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE

%colour defines whether using sift with colour or grayscale
%step defines the step size for sift

% image_paths is an N x 1 cell array of strings where each string is an
% image path on the file system.

% This function assumes that 'vocab.mat' exists and contains an N x 128
% matrix 'vocab' where each row is a kmeans centroid or visual word. This
% matrix is saved to disk rather than passed in a parameter to avoid
% recomputing the vocabulary every time at significant expense.

% image_feats is an N x d matrix, where d is the dimensionality of the
% feature representation. In this case, d will equal the number of clusters
% or equivalently the number of entries in each image's histogram.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be used for extra
  credit if you are constructing a "spatial pyramid".
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.
D = vl_alldist2(X,Y) 
   http://www.vlfeat.org/matlab/vl_alldist2.html
    returns the pairwise distance matrix D of the columns of X and Y. 
    D(i,j) = sum (X(:,i) - Y(:,j)).^2
    Note that vl_feat represents points as columns vs this code (and Matlab
    in general) represents points as rows. So you probably want to use the
    transpose operator '  You can use this to figure out the closest
    cluster center for every SIFT feature. You could easily code this
    yourself, but vl_alldist2 tends to be much faster.
%}

% load('vocab.mat')

imageNum = length(image_paths);

%0 level histogram
% level0_hist = zeros(size(vocab,2),1);

% MAX_LEVEL = 3;
% STEP = 3;
% COLORSPACE = "GRAYSCALE";
psize = VOCAB_SIZE*(4^(MAX_LEVEL+1)-1)/3;
image_feats = zeros(imageNum,psize);
% Read in images and construct sift histograms
for i =1 :imageNum 
    
    if(mod(i,200)==0)
       disp(i); 
    end
    img = imread(image_paths{i});
  
    
    if(COLORSPACE == "GRAYSCALE")
        img = rgb2gray(img);
        img = im2single(img);    
    elseif(COLORSPACE == "RGB")
        img = getRGB_VALUES(img);
    end   
    [size_y,size_x] = size(img);

     
    
    %SIFT_features = d*M where M is num of features sampled , d = 128.
    %locations 2*n list of locations
    %manipulate step, bin size, and smoothing parameters
    [locations, SIFT_features] = vl_dsift(img,'step',STEP_SIZE,'size',BIN_SIZE,'fast');
 
    %convert matrix to single precision
     SIFT_features = single(SIFT_features);
        
    %workout local cluster
    D = vl_alldist2(vocab,SIFT_features,DISTANCE);
       
    %assign local feature to nearest cluster center
    %min of each row of distances corresponds to closest
    [~,ind] = min(D);
            
    %build histograms
    

    %create array to store pyramid histograms     
    histograms_list = zeros(size(vocab,2),getHistogramCounter(MAX_LEVEL));
    

    % histogram for level 0
    % this is the histogram of the image with 1 quadrant

     level0_hist= zeros(size(vocab,2),1);
    for j =1 : size(ind,2)
        x = ind(j);
        level0_hist(x) = level0_hist(x)+1;
    end
    

    % register first histogram for level 0
    histograms_list(:,1) = level0_hist;
    histograms_list = build_pyramid_kernel(MAX_LEVEL,size_x,size_y,vocab,...
        locations,ind,histograms_list);
%%%
    

%     % normalise histogram by number of pixels
%     histograms_list = histograms_list/pixelsCounter;
      hist_listNorm = histograms_list - min(histograms_list(:));
      hist_listNorm = hist_listNorm ./ max(hist_listNorm(:));    
        
     weighted_hist_list = apply_weight(MAX_LEVEL,hist_listNorm);
%        
%         
%     %flatten to a 1d histogram
%     hist_list_vec = weighted_hist_list(:);
% 

%addd histogram to feature list
    hist_list_vec=weighted_hist_list(:);
    image_feats(i,:) = hist_list_vec;
end
%  image_feats = image_feats';

end


function output = build_pyramid_kernel(MAX_LEVEL,size_x,size_y,vocab,...
                            locations,ind,histograms_list)
    hist_index = 2; % 2 is the first level 1 histograms_list index          
    %level >0
    % LEVELS
    m_lvl = MAX_LEVEL-1;
    for k =0:m_lvl
%         fprintf("--level -> %d/%d \n",k,m_lvl);
		%how many quadrants at this level
		quadrantCounter = 2^(2*k);
		% each quadrant has its own histogram

		% find size of each cell
		q_size_x = floor(size_x/quadrantCounter) ;
		q_size_y = floor(size_y/quadrantCounter);
        
%         weight = getWeight(MAX_LEVEL,k);
		% QUADRANTS
        
        for q = 1 : quadrantCounter
%             fprintf("level:%d  weight:%.4f  q:%d \n",k,weight,q);
          %%%%%%%
            q_hist = check_for_match(q,q_size_x,q_size_y,locations,ind,vocab);
            
%             hist_temp = q_hist*weight;
            
            histograms_list(:,hist_index) = q_hist;%*weight;              

            hist_index = hist_index + 1;
        end
%        fprintf("-------------\n");
        
    end
    
    output = histograms_list;
end

function weight = getWeight(MAX_LEVEL,current_level)
     weight = 1/2^(MAX_LEVEL - current_level); 
end

function output = check_for_match(quad,q_size_x,q_size_y,locations,ind,vocab)
    hist = zeros(size(vocab,2),1);
    % loop through the whole quadrant and get histogram - could probably be vectorised
    
    x_max = quad * q_size_x;
    y_max = quad * q_size_y;
    
      for x = (quad-1)*q_size_x + 1 : x_max
          for y = (quad-1)*q_size_y + 1 : y_max
              % check if there is a feature at this location

              % check x
              x_loc = locations(1,:) == x;
              % check y
              y_loc = locations(2,:) == y;

              % check if there is an overlap
              xy_loc = bitand(x_loc,y_loc);

              xy_index = find(xy_loc);

              if(~isempty(xy_index))
                  % there is a feature
                  index = ind(xy_index);

                  % add to quadrant histogram
                  hist(index) = hist(index) + 1;
              end
          end
      end
      output = hist;
end

function output = apply_weight(MAX_LEVEL,histogram_list)

    % apply weightings based  (1/2^(L - l) L 
    index = 1;
    for level = 0 : MAX_LEVEL-1
        quadrants = 2^(2*level);
        
        weight = 1/2^(MAX_LEVEL - level+1); 
%          fprintf("LEVEL:%d WEIGHT:%d Quads: %d \n",level,weight,quadrants);
     
        if(level ~= 0)
            
            start_ind = index;
            end_ind = index + quadrants - 1;
%             fprintf("index %d - %d \n",index,end_ind);
%             fprintf("weight %d:%f \n",level,weight);
            histogram_list(:,start_ind:end_ind) = histogram_list(:,start_ind:end_ind) * weight;
        else
            weight = 1/2^MAX_LEVEL;
            
%             fprintf("index 1 \n");
%             fprintf("weight %d:%f \n",level,weight);
            histogram_list(:,1) = histogram_list(:,1) *  weight;
        end
        
        index = index + quadrants;
    end
    output = histogram_list;
    
end

function hist_counter = getHistogramCounter(MAX_LEVEL)
    hist_counter = 0;
    for z = 0 : MAX_LEVEL
        hist_counter = hist_counter + 2^(2*z);
    end
    
end