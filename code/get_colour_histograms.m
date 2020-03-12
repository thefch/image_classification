function img_feats = get_colour_histograms(q,input_paths,colour_space,USE_NORM,USE_MEAN)
    
    % init the output list
    switch(colour_space)
        case("GRAYSCALE")
            img_feats = zeros(size(input_paths,1),q*q);
        otherwise
            img_feats = zeros(size(input_paths,1),q*q*q);
    end
    
    for i=1 : length(input_paths)
        %fprintf("Colour Histogram:%d \n",i);
        path = input_paths{i};
        I = imread(path);
%        imshow(I);
%        disp(i);

        switch(colour_space)
            case("RGB")
                img = im2double(I);   
            case("HSV")
                converted_img = rgb2hsv(I);
                img = im2double(converted_img);
            case("GRAYSCALE")
                converted_img = rgb2gray(I);
                img = im2double(converted_img);
            case("YCBR")
                converted_img = rgb2ycbcr(I);
                img = im2double(converted_img);
            case("NTSC")
                converted_img = rgb2ntsc(I);
                
                % because some values are 0, we need to avoid that
                img = (converted_img + 1) / 2;
        end
        
        hist_1d = getHistogram(img,q,colour_space,USE_NORM,USE_MEAN);
        
        % each histogram is a row
        img_feats(i,:) = hist_1d;
        
       %if (mod(i, 100) == 0)
       %    fprintf("Completed training example: %d \n", i);
       %end
    end
end

% return the histogram, depending if its from a 3-value(3dmat) image or 1(2dmat)
function hist = getHistogram(img,q,colour_space,USE_NORM,USE_MEAN)
    imquant = round(img*(q-1)) + 1;
    
    if colour_space == "GRAYSCALE"
        hist = getHistFrom2Dmat(imquant,q);
    else
        hist = getHistFrom3Dmat(imquant,q,USE_NORM,USE_MEAN);
    end
end

% get color histogram for the images that are represented in 3D
% colorvalue array
function hist_1d = getHistFrom3Dmat(imquant,q,USE_NORM,USE_MEAN)
    [rows, columns, numberOfColorChannels] = size(imquant);
    hist = zeros(q,q,q);
    % generate histogram
    for j=1:rows
        for k=1:columns
            red = imquant(j,k,1);

            green= imquant(j,k,2);

            blue= imquant(j,k,3);
            

            %fprintf("%d %d %d \n",red,green,blue);
            hist(red,green,blue) = hist(red,green,blue)+1;
        end
    end
    %hist_1d = hist(:);
    
    % reset the first index for distracting pixels(noise)
    hist(1,1,1) = 0;    
    hist_1d = reshape(hist,1,[]);
    
    if(USE_MEAN)
        hist_1d = hist_1d - mean(hist_1d(:));
    end
 	if (USE_NORM)
 		hist_1d = hist_1d ./ mean(hist_1d(:));
 	end

    %hist_1d = hist_1d;
end

% get color histogram for all the images that are represented in 2D
% colorvalue array
function hist_1d = getHistFrom2Dmat(imquant,q)
    [rows, columns, numberOfColorChannels] = size(imquant);
    hist = zeros(q);
    for j=1:rows
        for k=1:columns
            c = imquant(j,k);
            hist(c) = hist(c)+1;
        end
    end
    
    hist_1d = reshape(hist,1,[]);
    
    %hist_1d = hist_1d;
end
