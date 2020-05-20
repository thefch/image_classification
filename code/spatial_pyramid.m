function image_feats = spatial_pyramid(image_paths,MAX_LEVEL,VOCAB_SIZE,DISTANCE,STEP_SIZE,COLORSPACE,BIN_SIZE,USE_PHOW,USE_NORM)

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

load('vocab.mat')

imageNum = length(image_paths);
psize = (VOCAB_SIZE*(4^(MAX_LEVEL+1)-1))/3;
image_feats = zeros(imageNum,psize);

%DISTANCE was removed, not much difference
%BIN_SIZE was removed, higher than 6 required too much computational power


for ii =1 :imageNum 
    % for debugging
%     if(mod(ii,200)==0)
%         %disp(ii); 
%     end
    I=imread(image_paths{ii});
    
    % extract sift features
    % use phow features if asked
    % or extract them with disft
    if (USE_PHOW)
       [locations, features] =get_phow(I,COLORSPACE,STEP_SIZE);
    else
        if(COLORSPACE == "GRAYSCALE")
            img = im2single(rgb2gray(I));  
        else
            img = get_color_values(I,COLORSPACE);

        end
     
    
        %SIFT_features = d*M where M is num of features sampled , d = 128.
        [locations, features] = vl_dsift(img,'step',STEP_SIZE);
         features = single(features); 
    end


    % normalze the histograms
    if(USE_NORM)
        tmp = sqrt(sum(features.^2, 2));
        features = features ./ repmat(tmp, [1 size(features,2)]);
     end


    %workout local cluster
      D = vl_alldist2(vocab',features);
      
     %assign local feature to nearest cluster center
     %min of each row of distances corresponds to closest
      [~,ind] = min(D);

    %create array to store pyramid histograms     
    histograms_list = zeros(VOCAB_SIZE,getHistogramCounter(MAX_LEVEL));
    

    % histogram for level 0
    % this is the histogram of the current image
    level0_hist= zeros(VOCAB_SIZE,1);
    % is doing it by considering the lowest distance of the features and
    % vocab as the original image.
    for j =1 : size(ind,2)
        x = ind(1,j);
         level0_hist(x,1) = level0_hist(x,1)+1;
    end

    %image's size
    [size_y,size_x] = size(I);
    
    % add the first histogram(level 0) to the histogram list
    histograms_list(:,1) = level0_hist;
    
    % build the pyramid of the current image
    histograms_list = build_pyramid_kernel(MAX_LEVEL,size_x,size_y,vocab,...
        locations,ind,histograms_list);

    % apply the weight of each level of the constructed pyramid
     weighted_hist_list = apply_weight(MAX_LEVEL,histograms_list);

    %addd histogram to feature list as a vector
    hist_list_vec=weighted_hist_list(:);
    
    % each row is an image's complete pyramid
    image_feats(ii,:) = hist_list_vec;
end

end


% build the pyramid of an image
function output = build_pyramid_kernel(MAX_LEVEL,size_x,size_y,vocab,...
                            locations,ind,histograms_list)
                    
    % 2 is the index of the first histogram that will be created
    % index 1 is occupied with level 0 histogram
    hist_index = 2; 
    
    % maximum level to be tested
    % -1 because level 0  is essentialy the first image and it does not
    % need any additional calculations when building the pyramid
    m_lvl = MAX_LEVEL-1;
    
    for k =1:m_lvl

		%number of quadrants at this level
        % each quadrant has its own histogram
		quadrantCounter = 2^(2*k);
		
		% find size of each cell
		q_size_x = floor(size_x/quadrantCounter) ;
		q_size_y = floor(size_y/quadrantCounter);
        
        % goes through all the quadrants of each level
        for q = 1 : quadrantCounter
             
            % check if the are any sift features in the current quadrant
            % returns a histogram with the sift features of that quadrant
            q_hist = check_for_match(q,q_size_x,q_size_y,locations,ind,vocab);
            
            % append that histogram to the histogram list
            % the histogram list holds all the histograms of the pyramid
            histograms_list(:,hist_index) = q_hist;        

            % holds the index of the current quadrant
            hist_index = hist_index + 1;
        end   
    end
    
    output = histograms_list;
end

% checks if a sift feature is inside a quadrant(quad)
% it basicly does a collision detection of a point and a square
function out = is_in_quadrant(j,locations,quad,q_size_x,q_size_y)
    %quadrant coordinates
    x_max = quad * q_size_x;
    y_max = quad * q_size_y;
    x_min = abs(x_max/quad-x_max);
    y_min = abs(y_max/quad-y_max);
    
    % holds the index of a sift feature from the original sift feature
    x=locations(1,j);
    y=locations(2,j);
    % if this x,y is in bounds with xmax,xmin,ymax,ymin then the current
    % quadrant contains a sift feature

    % simple collision detection to check if a point is in a square
    % here it is used to check if a spatial point is inside of a range of
    % values(quadrant)
    if (x > x_min && ... 
        x <= x_max && ...
        y > y_min && ...
        y <= y_max) 
    % collision detected
        out = true;
    else
        out = false;
        
    end
    
end

% returns a histogram count of sift features for a specific quadrant (q)
% loop through the whole quadrant and get histogram
function output = check_for_match(quad,q_size_x,q_size_y,locations,ind,vocab)

    %   get te hist of the image
    %   eg. hist(1:50,1:50) = create hist histcoutns()
    %   add it to the final hist of that level
    %   apply weight for the level
    %   cosntruct the final hist
    
    hist= zeros(size(vocab,1),1);
    for j =1 : size(ind,2)
        index=ind(1,j);
        
        % returns true if a sift features is in this quadrant
        if(is_in_quadrant(j,locations,quad,q_size_x,q_size_y))
            % increament the histogram count by 1 
            %if feature exists in quadrant
            hist(index,1) = hist(index,1)+1;
        end
    end
      output = hist;
end

% apply the weight of each level of a pyramid
function output = apply_weight(MAX_LEVEL,histogram_list)
    index = 1;
    for level = 0 : MAX_LEVEL-1
        %number of quadrants of each level
        quadrants = 2^(2*level);
        
        % weight value
        weight = 1/2^(MAX_LEVEL - level+1); 
        
        % apply the weight of each level
        if(level ~= 0)
            start_ind = index;
            end_ind = index + quadrants - 1;
            histogram_list(:,start_ind:end_ind) = histogram_list(:,start_ind:end_ind) * weight;
        else
            weight = 1/2^MAX_LEVEL;
            histogram_list(:,1) = histogram_list(:,1) *  weight;
        end
        
        % holds the index of each level
        index = index + quadrants;
    end
    output = histogram_list;
    
end

% returns the number of histograms that will be created for a pyramid with
% maximum level MAX_LEVEL
function hist_counter = getHistogramCounter(MAX_LEVEL)
    hist_counter = 0;
    for z = 0 : MAX_LEVEL
        hist_counter = hist_counter + 2^(2*z);
    end
    
end