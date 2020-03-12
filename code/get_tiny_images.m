function image_feats = get_small_images(im_size, input_paths,COLOUR_SPACE,USE_NORM,USE_MEAN)
    
    % init size and feats list
    N = size(input_paths, 1);
    image_feats = zeros(length(input_paths(:)), im_size*im_size);

    for i = 1:N    
        I = imread(input_paths{i});
        
        %convert image to selected color space
        switch(COLOUR_SPACE)
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
                %imshow(converted_img);
                %img = im2double(converted_img);
                %img = (converted_img + 1) / 2;
                img = im2double(converted_img);
        end
        
        resized = imresize(img, [im_size im_size]); 
        %imshow(resized);
        
        % add up each pixel's rgb value to a single index in the list
        temp = mean(resized,3); 
        %imshow(temp);
        
        % convert it into 1D vector
        image_feats(i,:) = reshape(temp,1,[]);
        
        if(USE_MEAN)
            image_feats(i,:) = image_feats(i,:) - mean(image_feats(i,:));
        end
        
        if(USE_NORM)
            image_feats(i,:) = image_feats(i,:) / norm(image_feats(i,:));  
        end
    end
end